#include "data_proc.h"
#include "omp.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace onlineclust;
using namespace cv;

int main(){

  DataProc test;
  try{
    Mat Image;
    char str[] = "desk_1_1.png";
    char type[] = "RGB";
    test.RGBD_reader(str, type,Image);
    //test.ShowImgDim(Image);
    //test.ImgShow(Image,"original");
    //cout << Image.step[0] *Image.step[1] << endl;
    // get sub-image
    //float centx = Image.rows/2;
    //float centy = Image.cols/2;
    Mat patch;
    test.im2patchMat(Image,{5,5},{1,1},patch);
    //test.ShowImgDim(patch);
    //Mat subImage;
    //getRectSubPix(patch, {5,5}, {30,10000}, subImage);
    //cout << subImage << endl;
    //test.ShowImgDim(subImage);
    //test.ImgShow(patch, "subimage");
    //cv::waitKey(0);


    Map<Matrix<int,Dynamic,Dynamic>, ColMajor, Stride<Dynamic,1> > im(reinterpret_cast<int*>(patch.data), patch.cols, patch.rows, Stride<Dynamic,1>(patch.rows,1));

    int array[24];
    for(int i = 0; i < 24; ++i) array[i] = i;
    cout << Map<MatrixXi, ColMajor, Stride<Dynamic,2> >
      (array, 3, 3, Stride<Dynamic,2>(8, 2))
    	 << endl;

  }
  catch(int e){};

  return 0;
}
