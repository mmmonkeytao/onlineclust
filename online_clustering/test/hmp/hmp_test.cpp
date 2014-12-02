#include "data_proc.h"
#include "omp.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace onlineclust;
using namespace cv;

int main(){

  DataProc test;
  OMP omp;
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
      // //test.ImgShow(Image,"original");
      
      // Mat patch;
      // test.im2patchMat(Image,{5,5},{1,1},patch);

      // test.ShowImgDim(patch);
      // cout << patch.at<double>(70,10000);
    }

    {
      // test sparse coding
      //test patch transformation
      Mat Image;
      char str[] = "cell_phone_1_1_6_crop.png";
      char type[] = "RGB";
      test.RGBD_reader(str,type,Image);
      
      test.ShowImgDim(Image);
      //test.ImgShow(Image,"original");
      
      Mat patch;
      Size stepsz = {5,5};
      Size psz = {5,5};
      test.im2patchMat(Image,psz,stepsz,patch);
      test.ShowImgDim(patch);
      
      MatrixXd D;
      omp.loadDct("rgbdevel_fulldic_1st_layer_5x5_crop.dat",75,150,D); 
 
      // convert Mat to MatrixXd
      Map<MatrixXd, RowMajor, Stride<1,Dynamic> > 
	im(reinterpret_cast<double*>(patch.data), patch.rows, patch.cols, Stride<1,Dynamic>(1,patch.cols));
      
      MatrixXd Gamma; //MatrixXd rand = MatrixXd::Zero(75,1); 
      //MatrixXd rand{75,1}; //rand.setRandom();
      omp.Batch_OMP(im, D, 10, Gamma);
      //cout << (rand - D*Gamma).norm()<<endl;
      //return 0;
      // reconstruct to original image 
      int pnw = ceil((double)Image.cols / (double)stepsz.width);
      int pnh = ceil((double)Image.rows / (double)stepsz.height);

      MatrixXd m = D*Gamma;
      Mat outMat{D.rows(), Gamma.cols(), CV_64FC1, m.data(), sizeof(double)*m.cols()};
      //Mat outMat{Image.rows, Image.cols, CV_8UC3, Image.data, sizeof(uint8_t)*Image.cols*3};
      Mat mat; 
     
      test.reconstructIm(outMat, psz.height, psz.width, pnh, pnw, mat);
      
      imwrite("cell_phone_1_1_6_crop_reconstruct.png", mat);
    }
	
  }
  catch(int e){};

  return 0;
}
