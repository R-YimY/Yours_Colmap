#include "ele_reconstruction.h"

colmap::CAEleRecond::CAEleRecond(
    const std::string &database_path,
    const std::string &image_folder,
    const std::string &mask_folder,
    const std::string &sparse_pose_folder,
    bool distortimg)
{

    database_ = database_path;
    image_folder_ = image_folder;
    mask_folder_ = mask_folder;
    ori_sparse_poses_folder_ = sparse_pose_folder;
    img_distort = distortimg;
    Init();
}

bool colmap::CAEleRecond::DenseReconElement()
{
    FeatureExtract();
    ExhaustiveMatch();
    PointTriangulate();
    ImageUndistorter();
    PatchMatchStereo();
    StereoFusion();
    return true;
}

/*/ ////////////////////////////////////////////////////////////////////////////
业务函数
// ///////////////////////////////////////////////////////////////////////////*/

bool colmap::CAEleRecond::Init()
{
    // 创建文件夹
    std::filesystem::path ori_sparse_path(ori_sparse_poses_folder_);
    std::filesystem::path parentPath = ori_sparse_path.parent_path();
    std::filesystem::path combined_tri_sparse_output = std::filesystem::path(parentPath) / tri_sparse_poses_folder_;
    std::filesystem::path combined_dense_output = std::filesystem::path(parentPath) / dense_output_folder_;

    tri_sparse_poses_folder_ = combined_tri_sparse_output;
    dense_output_folder_ = combined_dense_output;

    std::filesystem::path combined_dense_points_output = std::filesystem::path(dense_output_folder_) / dense_output_pointcloud_;
    std::filesystem::path combined_dense_points_output_json = std::filesystem::path(dense_output_folder_) / dense_output_pointcloud_json_;

    dense_output_pointcloud_ = combined_dense_points_output;
    dense_output_pointcloud_json_ = combined_dense_points_output_json;
    LOG(INFO) << dense_output_folder_ << std::endl;
    LOG(INFO) << dense_output_pointcloud_ << std::endl;

    return true;
}

bool colmap::CAEleRecond::FeatureExtract()
{
    std::string image_list_path;
    int camera_mode = -1;
    std::string descriptor_normalization = "l1_root";

    OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &camera_mode);
    options.AddDefaultOption("image_list_path", &image_list_path);
    options.AddDefaultOption("descriptor_normalization",
                             &descriptor_normalization,
                             "{'l1_root', 'l2'}");
    options.AddExtractionOptions();
    options.Check();
    // options.Parse(argc, argv);

    ImageReaderOptions reader_options = *options.image_reader;

    reader_options.database_path = database_;
    reader_options.image_path = image_folder_;
    reader_options.mask_path = mask_folder_;

    if (camera_mode >= 0)
    {
        UpdateImageReaderOptionsFromCameraMode(reader_options,
                                               (CameraMode)camera_mode);
    }

    StringToLower(&descriptor_normalization);
    if (descriptor_normalization == "l1_root")
    {
        options.sift_extraction->normalization =
            SiftExtractionOptions::Normalization::L1_ROOT;
    }
    else if (descriptor_normalization == "l2")
    {
        options.sift_extraction->normalization =
            SiftExtractionOptions::Normalization::L2;
    }
    else
    {
        LOG(ERROR) << "Invalid `descriptor_normalization`";
        return EXIT_FAILURE;
    }

    if (!image_list_path.empty())
    {
        reader_options.image_list = ReadTextFileLines(image_list_path);
        if (reader_options.image_list.empty())
        {
            return EXIT_SUCCESS;
        }
    }

    if (!ExistsCameraModelWithName(reader_options.camera_model))
    {
        LOG(ERROR) << "Camera model does not exist";
    }

    if (!VerifyCameraParams(reader_options.camera_model,
                            reader_options.camera_params))
    {
        return EXIT_FAILURE;
    }

    if (!VerifySiftGPUParams(options.sift_extraction->use_gpu))
    {
        return EXIT_FAILURE;
    }

    auto feature_extractor = CreateFeatureExtractorController(
        reader_options, *options.sift_extraction);

    if (options.sift_extraction->use_gpu && kUseOpenGL)
    {
        RunThreadWithOpenGLContext(feature_extractor.get());
    }
    else
    {
        feature_extractor->Start();
        feature_extractor->Wait();
    }

    return EXIT_SUCCESS;
}

bool colmap::CAEleRecond::ExhaustiveMatch()
{
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddExhaustiveMatchingOptions();
    options.Check();
    if (!VerifySiftGPUParams(options.sift_matching->use_gpu))
    {
        return EXIT_FAILURE;
    }

    auto matcher = CreateExhaustiveFeatureMatcher(*options.exhaustive_matching,
                                                  *options.sift_matching,
                                                  *options.two_view_geometry,
                                                  database_);

    if (options.sift_matching->use_gpu && kUseOpenGL)
    {
        RunThreadWithOpenGLContext(matcher.get());
    }
    else
    {
        matcher->Start();
        matcher->Wait();
    }

    return EXIT_SUCCESS;
}

bool colmap::CAEleRecond::PointTriangulate()
{
    std::string input_path;
    std::string output_path;

    bool clear_points = true;
    bool refine_intrinsics = false;
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption(
        "clear_points",
        &clear_points,
        "Whether to clear all existing points and observations and recompute "
        "the image_ids based on matching filenames between the model and the "
        "database");
    options.AddDefaultOption("refine_intrinsics",
                             &refine_intrinsics,
                             "Whether to refine the intrinsics of the cameras "
                             "(fixing the principal point)");
    options.AddMapperOptions();
    options.Check();

    input_path = ori_sparse_poses_folder_;
    output_path = std::string(tri_sparse_poses_folder_);
    CreateDirIfNotExists(output_path);

    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->Read(input_path);

    RunPointTriangulatorImpl(reconstruction,
                             database_,
                             image_folder_,
                             output_path,
                             *options.mapper,
                             clear_points,
                             refine_intrinsics);
    return EXIT_SUCCESS;
}

bool colmap::CAEleRecond::ImageUndistorter()
{
    std::string input_path;
    std::string output_path;
    std::string image_list_path;
    int num_patch_match_src_images = 20;
    CopyType copy_type;

    UndistortCameraOptions undistort_camera_options;

    OptionManager options;
    options.AddImageOptions();
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption("image_list_path", &image_list_path);
    options.AddDefaultOption("num_patch_match_src_images",
                             &num_patch_match_src_images);
    options.AddDefaultOption("blank_pixels",
                             &undistort_camera_options.blank_pixels);
    options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
    options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
    options.AddDefaultOption("max_image_size",
                             &undistort_camera_options.max_image_size);
    options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
    options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
    options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
    options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
    options.Check();

    // 配置地址
    input_path = tri_sparse_poses_folder_;
    output_path = dense_output_folder_;
    *options.image_path = image_folder_;

    CreateDirIfNotExists(output_path);

    PrintHeading1("Reading reconstruction");
    Reconstruction reconstruction;
    reconstruction.Read(input_path);
    LOG(INFO) << StringPrintf("=> Reconstruction with %d images and %d points",
                              reconstruction.NumImages(),
                              reconstruction.NumPoints3D());

    std::vector<image_t> image_ids;
    if (!image_list_path.empty())
    {
        const auto &image_names = ReadTextFileLines(image_list_path);
        for (const auto &image_name : image_names)
        {
            const Image *image = reconstruction.FindImageWithName(image_name);
            if (image != nullptr)
            {
                image_ids.push_back(image->ImageId());
            }
            else
            {
                LOG(WARNING) << "Cannot find image " << image_name;
            }
        }
    }

    copy_type = CopyType::COPY;
    std::unique_ptr<BaseController> undistorter;
    undistorter = std::make_unique<COLMAPUndistorter>(undistort_camera_options,
                                                      reconstruction,
                                                      *options.image_path,
                                                      output_path,
                                                      num_patch_match_src_images,
                                                      copy_type,
                                                      image_ids);

    undistorter->Run();
    return EXIT_SUCCESS;
}

bool colmap::CAEleRecond::PatchMatchStereo()
{

    std::string workspace_path;
    std::string workspace_format = "COLMAP";
    std::string pmvs_option_name = "option-all";
    std::string config_path;

    OptionManager options;
    options.AddRequiredOption(
        "workspace_path",
        &workspace_path,
        "Path to the folder containing the undistorted images");
    options.AddDefaultOption(
        "workspace_format", &workspace_format, "{COLMAP, PMVS}");
    options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
    options.AddDefaultOption("config_path", &config_path);
    options.AddPatchMatchStereoOptions();

    workspace_path = dense_output_folder_;
    options.patch_match_stereo->geom_consistency = true;

    StringToLower(&workspace_format);
    if (workspace_format != "colmap" && workspace_format != "pmvs")
    {
        LOG(ERROR) << "Invalid `workspace_format` - supported values are "
                      "'COLMAP' or 'PMVS'.";
        return EXIT_FAILURE;
    }

    mvs::PatchMatchController controller(*options.patch_match_stereo,
                                         workspace_path,
                                         workspace_format,
                                         pmvs_option_name,
                                         config_path);

    controller.Run();

    return EXIT_SUCCESS;
}

bool colmap::CAEleRecond::StereoFusion()
{
    std::string workspace_path;
    std::string input_type = "geometric";
    std::string workspace_format = "COLMAP";
    std::string pmvs_option_name = "option-all";
    std::string output_type = "PLY";
    std::string output_path;
    std::string bbox_path;
    OptionManager options;
    options.AddRequiredOption("workspace_path", &workspace_path);
    options.AddDefaultOption(
        "workspace_format", &workspace_format, "{COLMAP, PMVS}");
    options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
    options.AddDefaultOption(
        "input_type", &input_type, "{photometric, geometric}");
    options.AddDefaultOption("output_type", &output_type, "{BIN, TXT, PLY}");
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption("bbox_path", &bbox_path);
    options.AddStereoFusionOptions();
    options.Check();

    // 配置文件地址
    workspace_path = dense_output_folder_;
    output_path = dense_output_pointcloud_;
    options.stereo_fusion->mask_path = mask_folder_;

    StringToLower(&input_type);
    if (input_type != "photometric" && input_type != "geometric")
    {
        LOG(ERROR) << "Invalid input type - supported values are "
                      "'photometric' and 'geometric'.";
        return EXIT_FAILURE;
    }
    if (!bbox_path.empty())
    {
        std::ifstream file(bbox_path);
        if (file.is_open())
        {
            auto &min_bound = options.stereo_fusion->bounding_box.first;
            auto &max_bound = options.stereo_fusion->bounding_box.second;
            file >> min_bound(0) >> min_bound(1) >> min_bound(2);
            file >> max_bound(0) >> max_bound(1) >> max_bound(2);
        }
        else
        {
            LOG(WARNING) << "Invalid bounds path: \"" << bbox_path
                         << "\" - continuing without bounds check";
        }
    }

    mvs::StereoFusion fuser(*options.stereo_fusion,
                            workspace_path,
                            workspace_format,
                            pmvs_option_name,
                            input_type);

    fuser.Run();

    Reconstruction reconstruction;

    // read data from sparse reconstruction
    if (workspace_format == "colmap")
    {
        reconstruction.Read(JoinPaths(workspace_path, "sparse"));
    }

    // overwrite sparse point cloud with dense point cloud from fuser
    reconstruction.ImportPLY(fuser.GetFusedPoints());

    LOG(INFO) << "Writing output: " << output_path;

    // write output
    StringToLower(&output_type);
    if (output_type == "bin")
    {
        reconstruction.WriteBinary(output_path);
    }
    else if (output_type == "txt")
    {
        reconstruction.WriteText(output_path);
    }
    else if (output_type == "ply")
    {
        WriteTextPlyPoints(output_path, fuser.GetFusedPoints());
        WritePointsJsonWithVisibility(dense_output_pointcloud_json_,
                                      fuser.GetFusedPoints(),
                                      fuser.GetFusedPointsVisibility());
        mvs::WritePointsVisibility(output_path + ".vis",
                                   fuser.GetFusedPointsVisibility());
    }
    else
    {
        LOG(ERROR) << "Invalid `output_type`";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/*/ ////////////////////////////////////////////////////////////////////////////
 功能函数
 // ///////////////////////////////////////////////////////////////////////////*/

bool colmap::CAEleRecond::WritePointsJsonWithVisibility(const std::string &output_path,
                                                        const std::vector<PlyPoint> &points,
                                                        const std::vector<std::vector<int>> &visibility)
{

    if (points.size() != visibility.size())
    {
        LOG(ERROR) << "points size != visibility size";
        return false;
    }
    Json::Value result;
    result["allpoints"] = Json::Value(Json::arrayValue);
    size_t nums = points.size();
    for (int i(0); i < nums; i++)
    {
        const auto &pt = points[i];
        const auto &pt_vis = visibility[i];
        Json::Value out_pt;
        out_pt["x"] = pt.x;
        out_pt["y"] = pt.y;
        out_pt["z"] = pt.z;
        out_pt["nx"] = pt.nx;
        out_pt["ny"] = pt.ny;
        out_pt["nz"] = pt.nz;
        out_pt["r"] = pt.r;
        out_pt["g"] = pt.g;
        out_pt["b"] = pt.b;
        out_pt["vis_list"] = Json::Value(Json::arrayValue);
        for (const auto &cam_num : pt_vis)
        {
            out_pt["vis_list"].append(cam_num);
        }
        result["allpoints"].append(out_pt);
    }
    Json::StreamWriterBuilder builder;
    // 写入文件
    std::ofstream ofs(output_path);
    ofs << Json::writeString(builder, result);

    return true;
}

void colmap::CAEleRecond::UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions &options, CameraMode mode)
{
    switch (mode)
    {
    case CameraMode::AUTO:
        options.single_camera = false;
        options.single_camera_per_folder = false;
        options.single_camera_per_image = false;
        break;
    case CameraMode::SINGLE:
        options.single_camera = true;
        options.single_camera_per_folder = false;
        options.single_camera_per_image = false;
        break;
    case CameraMode::PER_FOLDER:
        options.single_camera = false;
        options.single_camera_per_folder = true;
        options.single_camera_per_image = false;
        break;
    case CameraMode::PER_IMAGE:
        options.single_camera = false;
        options.single_camera_per_folder = false;
        options.single_camera_per_image = true;
        break;
    }
}

bool colmap::CAEleRecond::VerifyCameraParams(const std::string &camera_model,
                                             const std::string &params)
{
    if (!ExistsCameraModelWithName(camera_model))
    {
        LOG(ERROR) << "Camera model does not exist";
        return false;
    }

    const std::vector<double> camera_params = CSVToVector<double>(params);
    const CameraModelId camera_model_id = CameraModelNameToId(camera_model);

    if (camera_params.size() > 0 &&
        !CameraModelVerifyParams(camera_model_id, camera_params))
    {
        LOG(ERROR) << "Invalid camera parameters";
        return false;
    }
    return true;
}

bool colmap::CAEleRecond::VerifySiftGPUParams(const bool use_gpu)
{
#if !defined(COLMAP_GPU_ENABLED)
    if (use_gpu)
    {
        LOG(ERROR)
            << "Cannot use Sift GPU without CUDA or OpenGL support; "
               "set SiftExtraction.use_gpu or SiftMatching.use_gpu to false.";
        return false;
    }
#endif
    return true;
}

void colmap::CAEleRecond::RunPointTriangulatorImpl(const std::shared_ptr<Reconstruction> &reconstruction, const std::string &database_path, const std::string &image_path, const std::string &output_path, const IncrementalPipelineOptions &options, const bool clear_points, const bool refine_intrinsics)
{
    THROW_CHECK_GE(reconstruction->NumRegImages(), 2)
        << "Need at least two images for triangulation";
    if (clear_points)
    {
        const Database database(database_path);
        reconstruction->DeleteAllPoints2DAndPoints3D();
        reconstruction->TranscribeImageIdsToDatabase(database);
    }

    auto options_tmp = std::make_shared<IncrementalPipelineOptions>(options);
    options_tmp->fix_existing_images = true;
    options_tmp->ba_refine_focal_length = refine_intrinsics;
    options_tmp->ba_refine_principal_point = false;
    options_tmp->ba_refine_extra_params = refine_intrinsics;

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(
        options_tmp, image_path, database_path, reconstruction_manager);
    mapper.TriangulateReconstruction(reconstruction);
    reconstruction->WriteText(output_path);
}
