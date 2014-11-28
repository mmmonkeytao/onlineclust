#include "data_proc.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include <opencv2/highgui/highgui.hpp>
#include <cstdint>
#include <cmath>

using namespace onlineclust;
using namespace cv;
using namespace std;

void DataProc::im2patchMat(Mat const& input, Size patchSize, Size stepSize,Mat &patch2dMat)
{
  int nPatchx = ceil((float)input.cols/(float)stepSize.width);
  int nPatchy = ceil((float)input.rows/(float)stepSize.height);
  patch2dMat = Mat{patchSize.width*patchSize.height*input.channels(),nPatchx * nPatchy, CV_16U};
  
  unsigned cols = 0;
  char nchannels = input.channels();
  int centx = 0, centy = 0;
  Mat patch = Mat{patchSize.width, patchSize.height, input.depth()};
  
  for(int j = 0; j < nPatchy; ++j){
    for(int i = 0; i < nPatchy; ++i){
      centx = i*stepSize.width;
      centy = j*stepSize.height;
      getRectSubPix(input, patchSize,{static_cast<float>(centx),static_cast<float>(centy)},patch,input.type());
      // copy to output matrix patch2dMat
      for(int k = 0; k < nchannels; ++k)
	for(int m = 0; m < patchSize.height; ++m)
	  for(int l = 0; l < patchSize.width; ++l){	     patch2dMat.at<unsigned>(l+m*patchSize.height+k*patchSize.width*patchSize.height, cols) = patch.at<unsigned>((l+k)*nchannels,m);
	  }
      ++cols;
    }
  }      
}


void DataProc::RGBD_reader(const char *file, char*type, Mat&matImage)
{
  cout << "\nReading " << type << " Image:\n" 
	    << file << endl; 
  
  if(!strcmp(type, "RGB")){
    matImage = imread(file,1);
    if(matImage.channels()!=3 || !matImage.data)throw runtime_error("\nError reading Image.\n");
    matImage.convertTo(matImage, CV_8U);
  } else if(!strcmp(type, "Depth")){
    matImage = imread(file, 0);
    if(matImage.channels()!=1)throw runtime_error("\nError reading Image\n");
    matImage.convertTo(matImage, CV_16U);
  }
  else {throw runtime_error("\nUnknown type of image.\n");}
}

void DataProc::ShowImgDim(Mat const&Image)const
{ 
  cout << "\nImage size:\n" 
       << Image.rows << "x" << Image.cols << "x" << Image.channels() << endl;
}

void DataProc::ImgShow(Mat const& Image, const char* name)const
{
    namedWindow( name, WINDOW_NORMAL ); // Create a window for display.
    imshow( name, Image );                // Show our image inside it.
    //waitKey(600); // Wait for a keystroke in the window
}
