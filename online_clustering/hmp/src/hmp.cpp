#include "hmp.h"
#include <Eigen/Core>
#include <cstring>
#include <stdexcept>


void onlineclust::HMP::load2Dcts(const char* layer1, const char*layer2, const char* type)
{
  if(!strcmp(type, "rgb")){
    loadDct(layer1, 75, 150, this->D1rgb);
    loadDct(layer2, 2400, 1000, this->D2rgb);
  } else {
    loadDct(layer1, 25, 75, this->D1depth);
    loadDct(layer2, 1200, 1000, this->D2depth);
  }
}

void onlineclust::HMP::hmp_core(Eigen::MatrixXd& X, const char* type, uint SPlevel[2], Eigen::MatrixXd& fea)
{
  uint nchannel = (!strcmp(type,"rgb"))? 3 : 1;
 
  //matSize imSize =  matSize(X.rows(), X.cols()/3);
  matSize gamma_sz =  matSize(ceil((float)(X.cols()/nchannel - (uint)(patchsz.width/2) * 2)/(float)stepsz.width1),ceil((float)(X.rows() - (uint)(patchsz.height/2) * 2)/(float)stepsz.height1));

  Eigen::MatrixXd patchMat;
  mat2patch(X, type, gamma_sz, patchMat);

  // remove dc part of signal
  char str[] = "column";
  remove_dc(patchMat, str);
  // 1st layer coding
  Eigen::MatrixXd Gamma; 
  
  if(nchannel == 3)
    omp::Batch_OMP(patchMat, D1rgb, SPlevel[0], Gamma);
  else 
    omp::Batch_OMP(patchMat, D1depth, SPlevel[0], Gamma);
 
  matSize psz1 = matSize(4,4);
  MaxPool_layer1_mode1(Gamma, psz1, gamma_sz);
  // 2nd layer learning
  if(nchannel == 3)
    omp::Batch_OMP(Gamma, D2rgb, SPlevel[1], fea);
  else
    omp::Batch_OMP(Gamma, D2depth, SPlevel[1], fea);
 
  matSize feaSize = matSize{gamma_sz.first/psz1.first, 
			    gamma_sz.second/psz1.second};
  uint pool[3] = {3,2,1};
  MaxPool_layer2(fea, feaSize, pool);
}

void onlineclust::HMP::MaxPool_layer2(Eigen::MatrixXd &fea, matSize const&feaSize, uint pool[3])
{ 
  uint rows = fea.rows();
  //uint cols = fea.cols();
  Eigen::MatrixXd temp = std::move(fea);
  uint nr = pow(pool[0],2) + pow(pool[1],2) + pow(pool[2],2);
  fea = Eigen::VectorXd::Zero(static_cast<int>(nr*rows),1);
  
  // pool 1
  Eigen::MatrixXd maxfea{rows, 16};
  uint psw = feaSize.first / pool[0];
  uint psh = feaSize.second / pool[0];
  uint blocksz = psw * psh;
  Eigen::MatrixXd max_tmp{rows, blocksz};
  
  for(uint j = 0; j < pool[0]; ++j )
    for(uint i = 0; i < pool[0]; ++i){
      uint spw = i * psw;
      uint sph = j * psh;
      
      for(uint l = 0; l < psh; ++l)
	max_tmp.block(0, l*psw, rows,psw) = temp.block(0,spw + (sph+l)*feaSize.first, rows, psw);

      fea.block((i+j*pool[0])*1000, 0, 1000,1) = max_tmp.rowwise().maxCoeff();
    }

  // pool 2
  psw = feaSize.first / pool[1];
  psh = feaSize.second / pool[1];
  blocksz = psw * psh;
  max_tmp = Eigen::MatrixXd{rows, blocksz};
  uint offset = pool[0] * pool[0];

  for(uint j = 0; j < pool[1]; ++j )
    for(uint i = 0; i < pool[1]; ++i){
      uint spw = i * psw;
      uint sph = j * psh;
      
      for(uint l = 0; l < psh; ++l)
	max_tmp.block(0, l*psw, rows,psw) = temp.block(0,spw + (sph+l)*feaSize.first, rows, psw);

      fea.block((offset+i+j*pool[1])*1000, 0, 1000,1) = max_tmp.rowwise().maxCoeff();
    }

  // pool 3
  uint offset2 = offset + pool[1]*pool[1];
  fea.block(offset2*1000, 0, 1000, 1) = fea.block(offset*1000, 0, 1000,1);
  for(uint j = 0; j < pool[1]; ++j )
    for(uint i = 1; i < pool[1]; ++i){
      fea.block(offset2*1000, 0, 1000, 1) = fea.block(offset2*1000, 0, 1000, 1).cwiseMax(fea.block((offset+i+j*pool[1])*1000, 0, 1000,1));
    }
  
  // normalize feature
  fea /= (fea.norm() + eps);

}

void onlineclust::HMP::MaxPool_layer1_mode1(Eigen::MatrixXd &Gamma, matSize const&psz, matSize const &realsz)
{
  uint feaSize = Gamma.rows();
  Eigen::MatrixXd temp = std::move(Gamma);
  
  uint nw = realsz.first / psz.first;
  uint nh = realsz.second / psz.second;

  Gamma = Eigen::MatrixXd{psz.first * psz.second * temp.rows(), nw * nh};
  
  uint spw = 0, sph = 0;
  for(uint j = 0; j < nh; ++j)
    for(uint i = 0; i < nw; ++i){
      spw = i * psz.first;
      sph = j * psz.second;
      
      for(uint l = 0; l < psz.second; ++l)
	for(uint m = 0; m < psz.first; ++m){
	  uint r = (m + l * psz.first) * feaSize;
	  uint fth = i + j*nw;
	  uint ith = (spw + m) + (sph + l) * realsz.first;  
	  Gamma.block( r,fth,feaSize,1) = temp.col(ith);
	}
    }
}

void onlineclust::HMP::mat2patch(Eigen::MatrixXd const& im, const char*type, matSize const& rsz, Eigen::MatrixXd &patchMat)
{
  uint nchannel;
  if(!strcmp(type, "rgb")){
    nchannel = 3;
  } else if (!strcmp(type, "depth")){
    nchannel = 1;
  } else {
    throw std::runtime_error("\nUnknow type!!!\n");
  }

  uint nPatchx = rsz.first; 
  uint nPatchy = rsz.second;

  patchMat = Eigen::MatrixXd{patchsz.width * patchsz.height * nchannel,
		   nPatchx * nPatchy};
  
  uint cols = 0, srow, scol;
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
