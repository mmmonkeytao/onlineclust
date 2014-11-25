#ifndef DATAPROC_H
#define DATAPROC_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

namespace onlineclust{

class DataProc{

public:

  DataProc(): depthImage(), rgbImage() {};
  ~DataProc(){};

  void RGBD_reader(const char*, const char*);
  void ImageVis() const;
  void ImgShow()const;

  const Mat& getRGBImg()const{ return rgbImage; };
  const Mat& getDepthImg()const{ return depthImage; };  
  void ShowImgDim()const;

private:

  Mat depthImage;
  Mat rgbImage;

  };

}

#endif

