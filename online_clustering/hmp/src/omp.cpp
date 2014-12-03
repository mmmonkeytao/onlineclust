#include "omp.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using namespace std;
using namespace onlineclust;


void OMP::im2patchMat(MatrixXd const& input, uint nchnl, uint psz[2], uint stepsz[2], MatrixXd &patch2dMat)
{
  uint npatchx = ceil((float)input.cols()/(float)stepsz[0]);
  uint npatchy = ceil((float)input.rows()/(float)stepsz[1]);
  patch2dMat = MatrixXd{nchnl*psz[0]*psz[1], npatchx*npatchy};
  
  for(uint k = 0; k < nchnl; ++k)
    for(int j = 0; j < input.rows(); ++j)
      for(int i = 0; i < input.cols(); ++i){
	
	  
	}
}

// void OMP::Batch_OMP( MatrixXd const& X, MatrixXd const& D, uint SPlevel, 
//  			      MatrixXd &Gamma )
// {
//   auto Xrow = X.rows(), Xcol = X.cols(), Drow = D.rows(), Dcol = D.cols();

//   if( !Xrow || !Xcol || !Drow || !Dcol || Xrow != Drow)
//     throw runtime_error("\nInput parameters are wrong in OMP::Bach_OMP.\n");

//   // compute matrix G = D' D, G: Dcol by Dcol
//   MatrixXd G = D.transpose() * D;
  
//   // compute apha^0 = D' x, alpha0: Dcol by Xcol
//   MatrixXd alpha0 = D.transpose() * X;   

//   // initialization 
//   Gamma = MatrixXd::Zero(Dcol, Xcol);
  
//   // iteration no. of obersations in X
//   for(auto i = 0; i < Xcol; ++i){
//     VectorXd alpha = alpha0.col(i);
//     // store max_k
//     vector<uint> I;
//     VectorXd gamma_I;
//     // inner loop for no. of atoms in D
//     for(uint j = 0; j < SPlevel; ++j){
//       uint k;
//       maxIdxVec(alpha, k);
//       I.push_back(k);
      
//       // retrieve G_II
//       MatrixXd G_II{j+1, j+1};
//       MatrixXd G_I{Dcol,j+1};
//       VectorXd a_I{j+1};
//       for(uint k1 = 0; k1 < I.size(); ++k1){
// 	a_I(k1) = alpha0(I[k1],i);
// 	G_I.col(k1) = G.col(I[k1]);
// 	for(uint k2 = 0; k2 < I.size(); ++k2){
	  
// 	  G_II(k2,k1) = G(I[k2],I[k1]);
// 	}
//       }
//       LLT<MatrixXd> llt{G_II};
//       gamma_I = llt.solve(a_I);
//       alpha = alpha0.col(i) - G_I * gamma_I;
//     }

//     for(auto k=0;k<I.size();++k)
//       Gamma(I[k],i) = gamma_I[k];
//   }
// }

void OMP::Batch_OMP( MatrixXd const& X, MatrixXd const& D, uint SPlevel, 
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
    // I stores ordered max_k
    vector<uint> I;    
    // initialize L
    MatrixXd L{1,1}; L << 1; 
    // initialize vector for if same atom is selected
    vector<bool> selected(Dcol,false);
    uint k;
    VectorXd a_I{1}, r_I{1};
    bool flag;
    VectorXd alpha = alpha0.col(i);
    // inner loop for no. of atoms in D
    for(uint j = 0; j < SPlevel; ++j){
      
      maxIdxVec(alpha, k);
      
      double alpha_k = alpha(k);
      if(selected[k] || alpha_k*alpha_k < 1e-14) break;
      
      if(j > 0){
	flag = true;
	updateL(L, G, I, k, flag);
	if(flag == false) break;
      }
      
      selected[k] = true;
      I.push_back(k);
      
      // retrieve sub-vector of alpha(i)
      if(j > 0){
	a_I.conservativeResize(j+1,1);
	r_I = VectorXd(j+1);
      }
      a_I(j) = alpha(k);
      LL_solver(L, a_I, "LL", r_I);
 
      // get G_I
      MatrixXd G_I{Dcol, j+1};
      uint counter = 0;
      for(auto &x: I){
	G_I.col(counter++) = G.col(x);
      }

      alpha = alpha0.col(i) - G_I * r_I;
    }

    uint count = 0;
    if(I.size() > 0)
      for(auto &x: I) 
	Gamma(x,i) = r_I(count++);
  }
}

inline void OMP::updateL( MatrixXd & L, MatrixXd const& G, 
			  vector<uint> const& I, uint k, bool& flag)
{
  if(static_cast<uint>(L.rows()) != I.size()) 
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

  double sum = 1 - omega.dot(omega);
  if(sum <= 1e-14){
    flag = false;
    return;
  }
  L(dim, dim) = sqrt(sum);
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

void OMP::loadDct(const char* file, int rows, int cols, MatrixXd& D)
{
  ifstream input(file);
  D = MatrixXd{rows, cols};

  cout << "Loading dictionary:\n";
  for(int j = 0; j < rows; ++j)
    for(int i = 0; i < cols; ++i)
      input >> D(j,i);

  cout << "Load completes. Dictionary size is: " << D.rows() << "x" << D.cols() << endl;
}


void OMP::maxIdxVec(VectorXd const& v, uint &maxIdx)
{
  double max = -1;
  
  for(int i = 0; i < v.size(); i++){
    if(fabs(v.coeff(i)) > max){
      max = fabs(v.coeff(i));
      maxIdx = i;
    }
  }
}


void OMP::remove_dc(MatrixXd &X, char* type)
{
  if(!strcmp(type, "column")){
    
    MatrixXd mean = MatrixXd::Zero(1,X.cols());
    
    for(int i = 0; i < X.rows(); ++i){
      mean += X.row(i);
    }
    mean /= (double)X.rows();

    for(int i = 0; i < X.rows(); ++i){
      X.row(i) -= mean;
    }
  } else {
    cerr << "\nUnknown type in OMP::remove_dc.\n";
  }
}
