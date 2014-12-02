#include "data_proc.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include <opencv2/highgui/highgui.hpp>
#include <cstdint>
#include <cmath>

using namespace onlineclust;
using namespace cv;
using namespace std;
using namespace Eigen;

typedef Vec<uint8_t, 3> Vec3u8;

void DataProc::im2patchMat(Mat const& input, Size psz, Size stpsz,Mat &patch2dMat)
{
  
  int nPatchx = ceil((float)(input.cols - psz.width)/(float)stpsz.width);
  int nPatchy = ceil((float)(input.rows - psz.height)/(float)stpsz.height);

  patch2dMat = Mat{psz.width * psz.height * input.channels(),
		   nPatchx * nPatchy, CV_64FC1};
  
  unsigned cols = 0, srow, scol;
  int npx = psz.height * psz.width;

  for(int j = 0; j < nPatchy; ++j){
    for(int i = 0; i < nPatchx; ++i){
      scol = i*stpsz.width;
      srow = j*stpsz.height;

      //getRectSubPix(input, psz,{static_cast<float>(scol),static_cast<float>(srow)},patch,input.type());
      // copy to output matrix patch2dMat, order r,g,b
	for(int m = 0; m < psz.height; ++m)
	  for(int l = 0; l < psz.width; ++l){	     
	    Vec3u8 v = input.at<Vec3u8>(srow + m, scol + l);
	    
	    patch2dMat.at<double>(l+m*psz.width,cols) = (double)v[2]/255.0;
	    patch2dMat.at<double>(l+m*psz.width + npx, cols) = (double)v[1]/255.0;
	    patch2dMat.at<double>(l+m*psz.width + 2*npx, cols) = (double)v[0]/255.0;
	  }
      ++cols;
    }
  }      
}

void DataProc::im2patchMat(Mat const& input, Size psz, Size stpsz,MatrixXd &patch2dMat)
{
  
  int nPatchx = ceil((float)(input.cols - psz.width)/(float)stpsz.width);
  int nPatchy = ceil((float)(input.rows - psz.height)/(float)stpsz.height);

  patch2dMat = MatrixXd{psz.width * psz.height * input.channels(),
		   nPatchx * nPatchy};
  
  unsigned cols = 0, srow, scol;
  int npx = psz.height * psz.width;

  for(int j = 0; j < nPatchy; ++j){
    for(int i = 0; i < nPatchx; ++i){
      scol = i*stpsz.width;
      srow = j*stpsz.height;

      //getRectSubPix(input, psz,{static_cast<float>(scol),static_cast<float>(srow)},patch,input.type());
      // copy to output matrix patch2dMat, order r,g,b
	for(int m = 0; m < psz.height; ++m)
	  for(int l = 0; l < psz.width; ++l){
	     
	    Vec3u8 v = input.at<Vec3u8>(srow + m, scol + l);	    
	    patch2dMat(l+m*psz.width,cols) = (double)v[2]/255.0;
	    patch2dMat(l+m*psz.width + npx, cols) = (double)v[1]/255.0;
	    patch2dMat(l+m*psz.width + 2*npx, cols) = (double)v[0]/255.0;
	  }
      ++cols;
    }
  }      
}

void DataProc::reconstructIm(Mat &im, Size imSize, Size patch, Size stepsz, Mat &mat)
{
  int imh = imSize.height, imw = imSize.width;
  int pszh = patch.height, pszw = patch.width;
  int pnw = ceil((double)(imw - pszw)/ (double)stepsz.width);
  int pnh = ceil((double)(imh - pszh)/ (double)stepsz.height);
  
  mat = Mat{pszh*pnh, pszw*pnw, CV_8UC3};

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

void DataProc::reconstructIm(MatrixXd &im, Size imSize, Size patch, Size stepsz, Mat &mat)
{
  int imh = imSize.height, imw = imSize.width;
  int pszh = patch.height, pszw = patch.width;
  int pnw = ceil((double)(imw - pszw)/ (double)stepsz.width);
  int pnh = ceil((double)(imh - pszh)/ (double)stepsz.height);
  
  mat = Mat{pszh*pnh, pszw*pnw, CV_8UC3};

  int psz = pszh * pszw;

  for(int j = 0; j < pnh; ++j)
    for(int i = 0; i < pnw; ++i){
      int sw = i * pszw;
      int sh = j * pszh;
      int pth = i + j*pnw;
  
      for(int l = 0; l < pszh; ++l)
	for(int k = 0; k < pszw; ++k){	  
	  Vec3u8 v;
	  v[2] = static_cast<uint8_t>(im(k+l*pszw, pth) * 255.0);
	  v[1] = static_cast<uint8_t>(im(k+l*pszw+psz, pth) * 255.0);
	  v[0] = static_cast<uint8_t>(im(k+l*pszw+psz*2, pth) * 255.0);
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

void DataProc::add_dc(MatrixXd& im)
{
  MatrixXd mean = MatrixXd::Zero(1, im.cols());

  for(int i = 0; i < im.rows(); ++i){
    mean += im.row(i);
  }
  
  mean /= (double)im.rows();

  for(int i = 0; i < im.rows(); ++i){
    im.row(i) += mean;
  }
}
