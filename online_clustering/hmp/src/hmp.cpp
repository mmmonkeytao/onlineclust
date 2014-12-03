#include "hmp.h"
#include "omp.h"
#include <Eigen/Core>
#include <cstring>
#include <stdexcept>

using namespace onlineclust;

void HMP::hmp_core(MatrixXd& X )
{
  MatrixXd D1rgb{75, 150};
  MatrixXd D2rgb{2400, 1000};
  
  OMP omp;
  omp.loadDct("rgbdevel_fulldic_1st_layer_5x5_crop.dat", 75, 150, D1rgb);
  omp.loadDct("rgbdeval_dic_2nd_layer_5x5_depthcrop.dat", 2400, 1000, D2rgb);

}

void HMP::MaxPool_layer1_BOMP(MatrixXd const&D, MatrixXd const&X, uint splevel, MatrixXd &SPcodes)
{
  


}

void HMP::mat2patch(MatrixXd const& im, const char*type, MatrixXd &patchMat)
{
  uint nchannel;
  if(!strcmp(type, "rgb")){
    nchannel = 3;
  } else if (!strcmp(type, "depth")){
    nchannel = 1;
  } else {
    throw runtime_error("\nUnknow type!!!\n");
  }
  
  uint nPatchx = ceil((float)(im.cols()/nchannel - patchsz.width)/(float)stepsz.width1);
  uint nPatchy = ceil((float)(im.rows() - patchsz.height)/(float)stepsz.height1);

  patchMat = MatrixXd{patchsz.width * patchsz.height * nchannel,
		   nPatchx * nPatchy};
  
  unsigned cols = 0, srow, scol;
  int npx = patchsz.height * patchsz.width;

  for(uint j = 0; j < nPatchy; ++j){
    for(uint i = 0; i < nPatchx; ++i){
      scol = i * stepsz.width1 * nchannel;
      srow = j * stepsz.height1;

      // copy to output matrix patch2dMat, order r,g,b
      for(uint ch = 0; ch < nchannel; ++ch)
	for(uint m = 0; m < patchsz.height; ++m)
	  for(uint l = 0; l < patchsz.width; ++l){
	    patchMat(l + m*patchsz.width + ch*npx, cols)
	      = im(srow + m, scol + l*nchannel + ch);
	  }
      ++cols;
    }
  }      

}
