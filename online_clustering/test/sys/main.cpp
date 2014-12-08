#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <chrono>
#include "pcloud.h"
#include "hmp.h"
#include "OnlineStarClustering.h"

using namespace std;
using namespace onlineclust;
using namespace std::chrono;

int main(){
  
  HMP hmpObj;
  OnlineStarClustering osc(0.5,14000);
  
  MatrixXd x{44,21};
  x.setRandom();
  uint splevel[2] = {5, 10};
  hmpObj.load2Dcts("dic_1st_layer_5x5_depth.dat", "dic_2nd_layer_5x5_depth.dat", "depth");

  // MatrixXd feature;
  // hmpObj.hmp_core(x,"depth",splevel, feature);
  // return 0;

  // load data from pcl
    // Initialization
  const uint nfiles = 10;
  // 
  PCloud Cloud;
  Cloud.load_pcloud("../data/", nfiles);

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

        //Cloud.proc_pcloud(pcloud, i);
        // seg plane model
	Cloud.plane_model_segmentation( Cloud.getCloud(i), pcloud);
        // clustering
        std::vector<pcl::PointIndices> cluster_indices;
        Cloud.euclidean_cluster_extraction(pcloud, cluster_indices);

        	/**
	 * convert range image to feature  
	 */
	pcl::RangeImage *range_image;
	uint rangeImage_num;

        Cloud.getRangeImage(pcloud, cluster_indices, range_image, rangeImage_num);
 	for(uint j = 0; j < rangeImage_num; ++j){
          MatrixXd fea;
 	  uint width = range_image[j].width;
 	  uint height = range_image[j].height;
         
          if(width > 20 && height > 20){
	    MatrixXf X{height, width};
	    X = Map<MatrixXf, Unaligned, Stride<1,Dynamic> >(range_image[j].getRangesArray(), height, width, Stride<1,Dynamic>(1,width));
          
	    X = X.cwiseMax(MatrixXf::Zero(height,width));
	    MatrixXd xnew = X.cast<double>();

	    high_resolution_clock::time_point t1 = high_resolution_clock::now();
	    hmpObj.hmp_core(xnew,"depth",splevel, fea);
	    high_resolution_clock::time_point t2 = high_resolution_clock::now();
	    cout << "Computed cloud: " << i << "\nrange image: " << j << endl
		 << "size :" << xnew.rows() << "x" << xnew.cols() << endl
		 << "HMP computation time(ms): " << duration_cast<milliseconds>(t2-t1).count() << endl;

	    cout << "Inserting into clustering:\n";
            //SparseVectorXd sparse = fea.sparseView();
            VectorXd sparse = fea.col(0);
            osc.insert(sparse);
            cout << "Clustering insertion completes.\n" << endl;
	  }

 	}

	// remove all points before adding
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(pcloud, "intensity");

       viewer.removeAllPointClouds();

	//viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3);
	//viewer.addPointCloudNormals<pcl::PointXYZI, pcl::Normal> (ocloud, cloud_normals, 80, 0.5, "normals");
	viewer.addPointCloud<pcl::PointXYZI>(pcloud, intensity_distribution, "KITTI Viewer");

	viewer.spinOnce();
	++i;

        delete[] range_image;

      } else {
	//viewer.spinOnce(100);
	break;
      }

      //boost::this_thread::sleep (boost::posix_time::microseconds (1000000));

    }

  osc.exportDot("overall.dot", false);
  cout << ".dot output completes." << endl; 
  return 0;
}
