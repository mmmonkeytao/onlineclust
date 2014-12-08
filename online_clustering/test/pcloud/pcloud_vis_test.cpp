#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "pcloud.h"

using namespace std;
using namespace onlineclust;

int main(int argc, char **argv){

  if(argc != 3){
    cout << "Usage: <./exec> <input_dir> <num_files>\n";
    return -1;
  }

  // Initialization
  const uint nfiles = atoi(argv[2]);
  // 
  PCloud Cloud;
  Cloud.load_pcloud(argv[1], nfiles);

  // create viewer
  pcl::visualization::PCLVisualizer viewer("KITTI Viewer");
  viewer.setBackgroundColor (0, 0, 0);

  //
  Cloud.initCamParam();
  viewer.setCameraParameters(Cloud.getCamera());
  viewer.updateCamera();

  // start processing points clouds
  uint i = 0;
  while (!viewer.wasStopped ()){

      if( i < nfiles ){
	// segmentation
	// output clouds
	_pclType1::Ptr pcloud ( new _pclType1);

	Cloud.proc_pcloud(pcloud, i);
	
	// remove all points before adding
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(pcloud, "intensity");
	viewer.removeAllPointClouds();

	//viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3);
	//viewer.addPointCloudNormals<pcl::PointXYZI, pcl::Normal> (ocloud, cloud_normals, 80, 0.5, "normals");
	viewer.addPointCloud<pcl::PointXYZI>(pcloud, intensity_distribution, "KITTI Viewer");
	viewer.spinOnce();
	++i;

      }
      else {
	viewer.spinOnce(100);
      }

      //boost::this_thread::sleep (boost::posix_time::microseconds (1000000));

    }

  return 0;
}

