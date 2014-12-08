#include "data_proc.h"
#include "hmp.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace onlineclust;
using namespace onlineclust::omp;
using namespace cv;
using namespace Eigen;
using namespace std;
using namespace data_proc;

int main(){

  try{

    // { // test r g b channel
    //   Mat Image;
    //   char str[] = "desk_1_1.png";
    //   char type[] = "RGB";
    //   test.RGBD_reader(str, type,Image);

    //   int rows = Image.rows, cols = Image.cols;
    //   typedef Vec<uint8_t, 3> Vec3u8;
    //   Mat R{rows,cols,CV_8UC3,Scalar::all(0)}, G{rows,cols,CV_8UC3,Scalar::all(0)}, B{rows,cols,CV_8UC3, Scalar::all(0)};

    //   for(int j = 0; j < cols; ++j)
    // 	for(int i = 0; i < rows; ++i){
    // 	  Vec3u8 v = Image.at<Vec3u8>(i,j);
    // 	  B.at<Vec3u8>(i,j) = Vec3u8{v[0],0,0};
    // 	  G.at<Vec3u8>(i,j) = Vec3u8{0,v[1],0};
    // 	  R.at<Vec3u8>(i,j) = Vec3u8{0,0,v[2]};
    // 	}	  

    //   imwrite("R.png",R);
    //   imwrite("G.png",G);
    //   imwrite("B.png",B);
    // }

    {
      // test ROI function
      // Mat Image;
      // char str[] = "desk_1_1.png";
      // char type[] = "RGB";
      // test.RGBD_reader(str, type,Image);

      // test.ShowImgDim(Image);
      // test.ImgShow(Image,"original");
      
      // Mat subImage;
      // getRectSubPix(Image, {40,40}, {30,30}, subImage);
      
      // test.ShowImgDim(subImage);
      // test.ImgShow(subImage, "subimage");
      // cv::waitKey(0);

    }

    {
      // test patch transformation
      // Mat Image;
      // char str[] = "desk_1_1.png";
      // char type[] = "RGB";
      // test.RGBD_reader(str, type,Image);

      // test.ShowImgDim(Image);
    //   // //test.ImgShow(Image,"original");
      
    //   // Mat patch;
    //   // test.im2patchMat(Image,{5,5},{1,1},patch);

    //   // test.ShowImgDim(patch);
    //   // cout << patch.at<double>(70,10000);
    // }

    // {
    //   // test sparse coding
    //   //test patch transformation
      Mat Image;
      char str[] = "desk_1_1.png";
      char type[] = "RGB";

      RGBD_reader(str, type,Image);
      ShowImgDim(Image);
      //test.ImgShow(Image,"original");
      
      MatrixXd patch;
      Size stepsz = {5,5};
      Size psz = {5,5};
      im2patchMat(Image,psz,stepsz,patch);
      cout << "\nPatch Matrix Size: " << patch.rows() << "x" << patch.cols() << endl;

      MatrixXd D;
      loadDct("rgbdevel_fulldic_1st_layer_5x5_crop.dat",75,150,D);  
      // convert Mat to MatrixXd
      //      Map<MatrixXd, RowMajor, Stride<1,Dynamic> > im(reinterpret_cast<double*>(patch.data), patch.rows, patch.cols, Stride<1,Dynamic>(1,patch.cols));
      
      MatrixXd Gamma;
      //omp.remove_dc(patch, "column");
      double mean = patch.mean();
      patch -= MatrixXd::Ones(patch.rows(), patch.cols()) * mean;
      Batch_OMP(patch, D, 5, Gamma); 

      // reconstruct to original image
      MatrixXd m = D * Gamma;
      Mat mat; 
      
      m += MatrixXd::Ones(patch.rows(), patch.cols()) * mean;      
      reconstructIm(m, Image.size(), psz, stepsz, mat);
      imwrite("cell_phone_1_1_6_crop_reconstruct.png", mat);
    }
	
  }
  catch(int e){};

  return 0;
}
