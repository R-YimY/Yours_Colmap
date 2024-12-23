#include "ele_reconstruction.h"

using namespace colmap;

int main(){


    std::string database_path = "/data/database.db";
    std::string image_folder= "/data/images";
    std::string mask_folder= "/data/mask";
    std::string sparse_pose_file= "/data/sparse";

    CAEleRecond EleReconder(database_path,image_folder,mask_folder,sparse_pose_file,true);
    EleReconder.DenseReconElement();
    return EXIT_SUCCESS;
}