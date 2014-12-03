#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "pcloud.h"

using namespace std;
using namespace onlineclust;

int main(int argc, char **argv)
{
    if(argc != 3){
        cout << "Usage: <./exec> <input_dir> <i-th_files>" << endl;
        return -1;
    }

    // Initialization
    uint ith_file = atoi(argv[2]);
    // 
    string infile(argv[1]);
    stringstream ss;
    ss << ith_file;
    infile = infile + "kitti." + ss.str() + ".pcd";
    PCloud Cloud;
    Cloud.load_pcloud(infile.c_str(), 1);

    _pclType1::Ptr pcloud(new _pclType1);

    // seg different normals
    //diff_normal_segmentation(cloud[i], ocloud, scale1, scale2, threshold, segradius, NUM_NEIGHBORS, STD_DEVIATION, radius, i);

    // seg plane model
    Cloud.plane_model_segmentation( Cloud.getCloud(0), pcloud);

    // clustering
    std::vector<pcl::PointIndices> cluster_indices;
    Cloud.euclidean_cluster_extraction(pcloud, cluster_indices);
    Cloud.vis_pointcloud2rangeimage(pcloud, cluster_indices);

    // calculate normals
    //pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    //cloud_normal(ocloud, cloud_normals, radius);

    // remove all points before adding
    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(ocloud, "intensity");
    //viewer.removeAllPointClouds();

    //viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3);
    //viewer.addPointCloudNormals<pcl::PointXYZI, pcl::Normal> (ocloud, cloud_normals, 80, 0.5, "normals");
    //viewer.addPointCloud<pcl::PointXYZI>(ocloud, intensity_distribution, "KITTI Viewer");
    //viewer.spinOnce();
      
    return 0;
}





