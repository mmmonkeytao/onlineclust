#ifndef DATAPROC_H
#define DATAPROC_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace onlineclust{

class DataProc{

public:

  DataProc(){};
  ~DataProc(){};

  // Note: specially for visualizing data set from
  // http://rgbd-dataset.cs.washington.edu/dataset/
  // convert image to Mat
  // file: file directory/name
  // type: "RGB" or "Depth"
  // matImage: Mat type of image
  void RGBD_reader(const char*file, char*type, cv::Mat&matImage);
  
  void ImgShow(cv::Mat const&, const char*)const;  
  void ShowImgDim(cv::Mat const&)const;
  
  /// image to patch matrix as output 
  /// in which each column represents patch from  
  /// every channel, the order is [r;g;b].
  /// 
  // template<typename T>
  //inline void img2patchMat(Mat const& input, Size patchSize, Size stepSize,Mat &outPatch2dMat);
  void im2patchMat(Mat const& input, Size patchSize, Size stepSize,Mat &patch2dMat);

  /// 
  ///
  /// @param im 
  ///
  void reconstructIm(Mat &im, int pszh, int pszw, int pnh, int pnw, Mat &out);

private:

  // Mat depthImage;
  // Mat rgbImage;

  };

}

#endif

