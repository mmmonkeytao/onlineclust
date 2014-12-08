#include "data_proc.h"
#include <opencv2/opencv.hpp>

using namespace onlineclust;
using namespace cv;
using namespace Eigen;
using namespace data_proc;

int main(){

  try{
    Mat Image;
    char str[] = "desk_1_1.png";
    char type[] = "RGB";
    RGBD_reader(str, type,Image);
    //test.ShowImgDim(Image);
    //test.ImgShow(Image,"original");
    //cout << Image.step[0] *Image.step[1] << endl;
    // get sub-image
    //float centx = Image.rows/2;
    //float centy = Image.cols/2;
    Mat patch;
    im2patchMat(Image,{5,5},{1,1},patch);
    //test.ShowImgDim(patch);
    //Mat subImage;
    //getRectSubPix(patch, {5,5}, {30,10000}, subImage);
    //cout << subImage << endl;
    //test.ShowImgDim(subImage);
    //test.ImgShow(patch, "subimage");
    //cv::waitKey(0);
    //MatrixXi im{patch.rows,patch.cols};
    
    //Eigen::Map<MatrixXi,RowMajor, Stride<1,Dynamic> > im(patch.data, patch.rows, patch.cols, Stride<1,Dynamic>(1, patch.rows));

    
  }
  catch(int e){};

  return 0;
}
