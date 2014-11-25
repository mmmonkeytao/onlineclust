#include "data_proc.h"
using namespace onlineclust;

int main(){

  DataProc test;
  try{
  test.RGBD_reader("apple_1_1_1_crop.png", "apple_1_1_1_depthcrop.png");
  test.ShowImgDim();
  test.ImgShow();
  }
  catch(int e){};

  return 0;
}
