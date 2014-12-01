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
  typedef Vec<uint8_t, 3> Vec3u8;

  int nPatchx = ceil((float)input.cols/(float)stepSize.width);
  int nPatchy = ceil((float)input.rows/(float)stepSize.height);

  patch2dMat = Mat{patchSize.width * patchSize.height * input.channels(),
		   nPatchx * nPatchy, CV_64FC1};
  
  unsigned cols = 0;
  //  char nchannels = input.channels();
  int srow = 0, scol = 0;
  int chsize = patchSize.height * patchSize.width;
  Mat patch = Mat{patchSize.height, patchSize.width, input.type()};
  
  for(int j = 0; j < nPatchy; ++j){
    for(int i = 0; i < nPatchx; ++i){
      scol = i*stepSize.width;
      srow = j*stepSize.height;

      getRectSubPix(input, patchSize,{static_cast<float>(scol),static_cast<float>(srow)},patch,input.type());
      // copy to output matrix patch2dMat, order r,g,b
	for(int m = 0; m < patch.rows; ++m)
	  for(int l = 0; l < patch.cols; ++l){	     
	    Vec3u8 v = patch.at<Vec3u8>(m,l);
	    patch2dMat.at<double>(l+m*patchSize.width,cols) = (double)v[2]/255.0;
	    patch2dMat.at<double>(l+m*patchSize.width + chsize, cols) = (double)v[1]/255.0;
	    patch2dMat.at<double>(l+m*patchSize.width + 2*chsize, cols) = (double)v[0]/255.0;
	  }
      ++cols;
    }
  }      
}

void DataProc::reconstructIm(Mat &im, int pszh, int pszw, int pnh, int pnw, Mat &mat)
{
  mat = Mat{pszh*pnh, pszw*pnw, CV_8UC3};
  typedef Vec<uint8_t, 3> Vec3u8;
  int psz = pszh * pszw;

  for(int j = 0; j < pnh; ++j)
    for(int i = 0; i < pnw; ++i){
      int sw = i * pszw;
      int sh = j * pszh;
      int pth = i + j*pnw;
  
      for(int l = 0; l < pszh; ++l)
	for(int k = 0; k < pszw; ++k){	  
	  Vec3u8 v;
	  v[2] = static_cast<uint8_t>(im.at<double>(k+l*pszw, pth) * 255);
	  v[1] = static_cast<uint8_t>(im.at<double>(k+l*pszw+psz, pth) * 255);
	  v[0] = static_cast<uint8_t>(im.at<double>(k+l*pszw+psz*2, pth) * 255);
	  
	  mat.at<Vec3u8>(sh+l,sw+k) = v;

	} //break;
 
    }

}

void DataProc::RGBD_reader(const char *file, char*type, Mat&matImage)
{
  cout << "\nReading " << type << " Image:\n" 
	    << file << endl; 
  
  if(!strcmp(type, "RGB")){
    
    matImage = imread(file,CV_LOAD_IMAGE_COLOR);
    if(matImage.channels()!=3 || !matImage.data)
      throw runtime_error("\nError reading Image.\n");

    matImage.convertTo(matImage, CV_8U); //8 bit unsigned

  } else if(!strcmp(type, "Depth")){

    matImage = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

    if(matImage.channels()!=1)
      throw runtime_error("\nError reading Image\n");

    matImage.convertTo(matImage, CV_16U); // 16bit unsigned
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
