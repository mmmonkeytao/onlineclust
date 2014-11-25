#include "data_proc.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include <opencv2/highgui/highgui.hpp>
#include <cstdint>

using namespace onlineclust;
using namespace cv;
using namespace std;

void DataProc::RGBD_reader(const char *fnRGB, const char *fnDepth)
{
  cout << "\nReading RGB and Depth Image: \n" 
	    << fnRGB << "\n" << fnDepth << endl; 
  
  rgbImage = imread(fnRGB, CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);
  depthImage = imread(fnDepth, CV_LOAD_IMAGE_ANYDEPTH);

  if(!rgbImage.data || !depthImage.data) throw runtime_error("Can't open RGB or depth file.\n");

  rgbImage.convertTo(rgbImage, IPL_DEPTH_8U);
  depthImage.convertTo(depthImage, IPL_DEPTH_16U);
}

void DataProc::ShowImgDim()const
{ 
  cout << "\nImage size:\n" 
       << "RGB: "<< rgbImage.rows << "x" << rgbImage.cols << "x" << rgbImage.channels() << endl 
       << "Depth: "<< depthImage.rows << "x" << depthImage.cols << "x" << depthImage.channels() << endl;
}

void DataProc::ImgShow()const
{
    namedWindow( "RGB Display window", WINDOW_NORMAL ); // Create a window for display.
    imshow( "RGB Display window", rgbImage );                // Show our image inside it.

    namedWindow( "Depth Display window", WINDOW_NORMAL ); // Create a window for display.
    imshow( "Depth Display window", depthImage );                // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window          
}
