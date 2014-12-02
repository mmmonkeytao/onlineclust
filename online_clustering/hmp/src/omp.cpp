#include "omp.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using namespace std;
using namespace onlineclust;


void OMP::im2patchMat(MatrixXd const& input, unsigned nchnl, unsigned psz[2], unsigned stepsz[2], MatrixXd &patch2dMat)
{
  unsigned npatchx = ceil((float)input.cols()/(float)stepsz[0]);
  unsigned npatchy = ceil((float)input.rows()/(float)stepsz[1]);
  patch2dMat = MatrixXd{nchnl*psz[0]*psz[1], npatchx*npatchy};

  for(unsigned k = 0; k < nchnl; ++k)
    for(int j = 0; j < input.rows(); ++j)
      for(int i = 0; i < input.cols(); ++i)
	{
	  
	}
}

void OMP::loadDct(const char* file, int rows, int cols, MatrixXd& D)
{
  ifstream input(file);
  D = MatrixXd{rows, cols};

  cout << "Loading dictrionary:\n";
  for(int j = 0; j < rows; ++j)
    for(int i = 0; i < cols; ++i)
      input >> D(j,i);

  cout << "Load completes. Dictionary size is: " << D.rows() << "x" << D.cols() << endl;
}

void OMP::Batch_OMP( MatrixXd const& X, MatrixXd const& D, unsigned SPlevel, 
 			      MatrixXd &Gamma )
{
  auto Xrow = X.rows(), Xcol = X.cols(), Drow = D.rows(), Dcol = D.cols();

  if( !Xrow || !Xcol || !Drow || !Dcol || Xrow != Drow)
    throw runtime_error("\nInput parameters are wrong in OMP::Bach_OMP.\n");

  // compute matrix G = D' D, G: Dcol by Dcol
  MatrixXd G = D.transpose() * D;
  
  // compute apha^0 = D' x, alpha0: Dcol by Xcol
  MatrixXd alpha0 = D.transpose() * X;   

  // initialization 
  Gamma = MatrixXd::Zero(Dcol, Xcol);
  
  // iteration no. of obersations in X
  for(auto i = 0; i < Xcol; ++i){
    VectorXd alpha = alpha0.col(i);
    // store max_k
    vector<unsigned> I;
    VectorXd gamma_I;
    // inner loop for no. of atoms in D
    for(unsigned j = 0; j < SPlevel; ++j){
      unsigned k;
      maxIdxVec(alpha, k);
      I.push_back(k);
      
      // retrieve G_II
      MatrixXd G_II{j+1, j+1};
      MatrixXd G_I{Dcol,j+1};
      VectorXd a_I{j+1};
      for(unsigned k1 = 0; k1 < I.size(); ++k1){
	a_I(k1) = alpha0(I[k1],i);
	G_I.col(k1) = G.col(I[k1]);
	for(unsigned k2 = 0; k2 < I.size(); ++k2){
	  
	  G_II(k2,k1) = G(I[k2],I[k1]);
	}
      }
      LLT<MatrixXd> llt{G_II};
      gamma_I = llt.solve(a_I);
      alpha = alpha0.col(i) - G_I * gamma_I;
    }

    for(auto k=0;k<I.size();++k)
      Gamma(I[k],i) = gamma_I[k];
  }
}

// void OMP::Batch_OMP( MatrixXd const& X, MatrixXd const& D, unsigned SPlevel, 
// 			      MatrixXd &Gamma )
// {
//   auto Xrow = X.rows(), Xcol = X.cols(), Drow = D.rows(), Dcol = D.cols();

//   if( !Xrow || !Xcol || !Drow || !Dcol || Xrow != Drow)
//     throw runtime_error("\nInput parameters are wrong in OMP::Bach_OMP.\n");

//   // compute matrix G = D' D, G: Dcol by Dcol
//   auto G = D.transpose() * D;

//   // compute apha^0 = D' x, alpha0: Dcol by Xcol
//   MatrixXd alpha0 = D.transpose() * X;   
//   auto alpha = alpha0;

//   // initialization 
//   MatrixXd L{1,1}; L << 1; 
//   Gamma = MatrixXd::Zero(Dcol, Xcol);

//   // iteration no. of obersations in X
//   for(auto i = 0; i < Xcol; ++i){
//     // I stores ordered max_k
//     vector<unsigned> I;
//     unsigned j = 0; 
//     // find index k at ith col of X which Argmax_k{|a_k|}
//     unsigned k;
//     maxIdxVec(alpha.col(i), k);
//     // a_I sub-vector of alpha0, r_I sub-vector of Gamma(k,i)
//     VectorXd a_I{1}, r_I{1};
//     r_I << alpha0(k,i);
//     a_I = r_I;
//     I.push_back(k);
//     alpha.col(i) = alpha0.col(i) - G.col(k) * r_I;
//     ++j;

//     // inner loop for no. of atoms in D
//     for(;j < SPlevel; ++j){
//       maxIdxVec(alpha.col(i), k);
//       // 
//       updateL(L, G, I, k);
//       I.push_back(k);
      
//       // retrieve sub-vector of alpha(i)
//       a_I.conservativeResize(j+1,1);
//       a_I(j) = alpha(k,i);
//       r_I = VectorXd(j+1);
//       LL_solver(L, a_I, "LL", r_I);
      
//       // get G_I
//       MatrixXd G_I{Dcol, j+1};
//       unsigned counter = 0;
//       for(auto &x: I){
// 	G_I.col(counter) = G.col(x);
// 	++counter;
//       }

//       alpha.col(i) = alpha0.col(i) - G_I * r_I;
//     }
//     unsigned count = 0;
//     for(auto &x: I)Gamma(x,i) = r_I(count++);
//   }
// }


inline void OMP::maxIdxVec(VectorXd const& v, unsigned &maxIdx)
{
  double max = -1;
  
  for(auto i = 0; i < v.size(); i++){
    if(fabs(v.coeff(i)) > max){
      max = fabs(v.coeff(i));
      maxIdx = i;
    }
  }
}

inline void OMP::updateL( MatrixXd & L, MatrixXd const& G, 
				   vector<unsigned> const& I, unsigned k )
{
  if(static_cast<unsigned>(L.rows()) != I.size()) 
    throw runtime_error("\nInput dimensions do not match(updateL).\n");

  auto dim = L.cols();
  VectorXd g{dim};

  for(auto i = 0; i < dim; ++i)
    g(i) = G(k, I[i]);

  // solve for w linear system L * w = g using Cholesky decomposition
  VectorXd omega{dim};
  LL_solver(L, g, "L", omega);
  //cout << omega.transpose() << endl;
  // update L = [ L 0; w^T sqrt(1 - w'w)]
  L.conservativeResize(dim+1, dim+1);
  L.bottomLeftCorner(1,dim) = omega.transpose();
  L.rightCols(1) = MatrixXd::Zero(dim+1,1);

  L(dim, dim) = sqrt(1 - omega.dot(omega));
}

inline void OMP::LL_solver(MatrixXd const& L, VectorXd const& b, const char* type, VectorXd& x)
{
  if(L.cols() != b.size() || !x.size()) 
    throw runtime_error("\nInput dimension errors(LL_solver).\n");

   VectorXd w{b.size()}, b1{b};
   
   // L^T * w = b/L
   for(auto i = 0; i < b.size(); ++i){

     for(auto j = 0; j < i; ++j)
       b1(i) -= w(j)*L(i,j);
       
     w(i) = (double)b1(i)/L(i,i);
   }

   if(!strcmp("L", type)) { x = w; return; }

   // w = x/L^T
   for(int i = b.size()-1; i >= 0; --i){

     for(int j = b.size()-1; j > i; --j)
       w(i) -= x(j)*L(j,i);
        
     x(i) = (double)w(i)/L(i,i);
   }
}
