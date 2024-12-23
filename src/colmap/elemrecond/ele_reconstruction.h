#pragma once
#include <iostream>
#include <filesystem>
#include <fstream>
#include <jsoncpp/json/json.h>
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/image/undistortion.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/fusion.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/logging.h"

enum class CameraMode
{
    AUTO = 0,
    SINGLE = 1,
    PER_FOLDER = 2,
    PER_IMAGE = 3
};

namespace colmap
{
#if defined(COLMAP_CUDA_ENABLED) || !defined(COLMAP_GUI_ENABLED)
    const bool kUseOpenGL = false;
#else
    const bool kUseOpenGL = true;
#endif

    class CAEleRecond
    {
    public:
        CAEleRecond() {};
        CAEleRecond(const std::string &database_path,
                    const std::string &image_folder,
                    const std::string &mask_folder,
                    const std::string &sparse_pose_file,
                    bool distortimg = false);

        bool DenseReconElement();

    private:
        bool Init();
        bool FeatureExtract();
        bool ExhaustiveMatch();
        bool PointTriangulate();
        bool ImageUndistorter();
        bool PatchMatchStereo();
        bool StereoFusion();

    private:
        bool img_distort = false;
        std::string database_ = "";
        std::string image_folder_ = "";
        std::string mask_folder_ = "";
        std::string ori_sparse_poses_folder_ = "";
        std::string tri_sparse_poses_folder_ = "tri_sparse";
        std::string dense_output_folder_ = "dense";
        std::string dense_output_pointcloud_ = "dense_pointcloud.ply";
        std::string dense_output_pointcloud_json_ = "dense_pointcloud.json";

    private:
        bool WritePointsJsonWithVisibility(const std::string &output_path,
                                           const std::vector<PlyPoint> &points,
                                           const std::vector<std::vector<int>> &visibility);
        void UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions &options,
                                                    CameraMode mode);
        bool VerifyCameraParams(const std::string &camera_model,
                                const std::string &params);
        bool VerifySiftGPUParams(const bool use_gpu);
        inline void RunThreadWithOpenGLContext(Thread *thread) {}

        void RunPointTriangulatorImpl(
            const std::shared_ptr<Reconstruction> &reconstruction,
            const std::string &database_path,
            const std::string &image_path,
            const std::string &output_path,
            const IncrementalPipelineOptions &options,
            const bool clear_points,
            const bool refine_intrinsics);
    };

}