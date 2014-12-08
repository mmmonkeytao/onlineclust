#include "hmp.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>


void onlineclust::omp::Batch_OMP( Eigen::MatrixXd const& X, Eigen::MatrixXd const& D, uint SPlevel, 
			      Eigen::MatrixXd &Gamma )
{
  auto Xrow = X.rows(), Xcol = X.cols(), Drow = D.rows(), Dcol = D.cols();

  if( !Xrow || !Xcol || !Drow || !Dcol || Xrow != Drow)
    throw std::runtime_error("\nInput parameters are wrong in OMP::Bach_OMP.\n");

  // compute matrix G = D' D, G: Dcol by Dcol
  Eigen::MatrixXd G = D.transpose() * D;
  
  // compute apha^0 = D' x, alpha0: Dcol by Xcol
  Eigen::MatrixXd alpha0 = D.transpose() * X;   
  
  // initialization 
  Gamma = Eigen::MatrixXd::Zero(Dcol, Xcol);
  
  // iteration no. of obersations in X
  for(auto i = 0; i < Xcol; ++i){
    // I stores ordered max_k
    std::vector<uint> I;    
    // initialize L
    Eigen::MatrixXd L{1,1}; L << 1; 
    // initialize     std::vector for if same atom is selected
        std::vector<bool> selected(Dcol,false);
    uint k;
    Eigen::VectorXd a_I{1}, r_I{1};
    bool flag;
    Eigen::VectorXd alpha = alpha0.col(i);
    // inner loop for no. of atoms in D
    for(uint j = 0; j < SPlevel; ++j){
      
      maxIdxVec(alpha, k);
      
      double alpha_k = alpha(k);
      if(selected[k] || alpha_k*alpha_k < 1e-14) break;
      
      if(j > 0){
	flag = true;
	onlineclust::omp::updateL(L, G, I, k, flag);
	if(flag == false) break;
      }
      
      selected[k] = true;
      I.push_back(k);
      
      // retrieve sub-    std::vector of alpha(i)
      if(j > 0){
	a_I.conservativeResize(j+1,1);
	r_I = Eigen::VectorXd(j+1);
      }
      a_I(j) = alpha(k);
      LL_solver(L, a_I, "LL", r_I);
 
      // get G_I
      Eigen::MatrixXd G_I{Dcol, j+1};
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

void onlineclust::omp::updateL( Eigen::MatrixXd & L, Eigen::MatrixXd const& G, 
	          std::vector<uint> const& I, uint k, bool& flag)
{
  if(static_cast<uint>(L.rows()) != I.size()) 
    throw std::runtime_error("\nInput dimensions do not match(updateL).\n");

  auto dim = L.cols();
      Eigen::VectorXd g{dim};

  for(auto i = 0; i < dim; ++i)
    g(i) = G(k, I[i]);

  // solve for w linear system L * w = g using Cholesky decomposition
      Eigen::VectorXd omega{dim};
  LL_solver(L, g, "L", omega);
  //cout << omega.transpose() << endl;
  // update L = [ L 0; w^T sqrt(1 - w'w)]
  L.conservativeResize(dim+1, dim+1);
  L.bottomLeftCorner(1,dim) = omega.transpose();
  L.rightCols(1) = Eigen::MatrixXd::Zero(dim+1,1);

  double sum = 1 - omega.dot(omega);
  if(sum <= 1e-14){
    flag = false;
    return;
  }
  L(dim, dim) = sqrt(sum);
}

void onlineclust::omp::LL_solver(Eigen::MatrixXd const& L, Eigen::VectorXd const& b, const char* type,     Eigen::VectorXd& x)
{
  if(L.cols() != b.size() || !x.size()) 
    throw std::runtime_error("\nInput dimension errors(LL_solver).\n");

       Eigen::VectorXd w{b.size()}, b1{b};
   
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

void onlineclust::omp::maxIdxVec(Eigen::VectorXd const& v, uint &maxIdx)
{
  double max = -1;
  
  for(uint i = 0; i < v.size(); i++){
    if(fabs(v.coeff(i)) > max){
      max = fabs(v.coeff(i));
      maxIdx = i;
    }
  }
}

