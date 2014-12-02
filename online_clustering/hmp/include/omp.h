///
/// @file   omp.h
/// @author Ye Tao
/// @date   SEP 21 18:38:58 2014
/// 
/// @brief  
/// Orthogonal Matching Pursuit Method 
/// To solve: min  |X - D*GAMMA|_2     s.t.  |GAMMA|_0 <= T
///
/// Initialization: I = ∅, α^0 = D'* , G = D'* D, and x = 0
/// 
/// Reference: [1]  Ron Rubinstein, Michael Zibulevsky and Michael Elad,
///                 "Efficient Implementation of the K-SVD Algorithm 
///                 using Batch Orthogonal Matching Pursuit". 
///                 Technical Report - CS Technion, April 2008. 

#ifndef OMP_H
#define OMP_H

#include <Eigen/Core>
#include <vector>

using namespace Eigen;
using namespace std;

namespace onlineclust{

 
class OMP{

 public:
    OMP(){};
   ~OMP(){};

  /// @param X : observation containing column signals
  /// @param Dct : columns normalized dictionary
  /// @param SPlevel : sparse level constrain
  ///
  /// @output: Sparse code GAMMA such that X ≈ D*GAMMA
  ///
  void Batch_OMP( MatrixXd const& X, MatrixXd const& Dct, unsigned SPlevel, MatrixXd& Gamma);

  /// 
  /// Modified from OMP-Box v10 Implementation of the Batch-OMP 
  /// and OMP-Cholesky algorithms for quick sparse-coding of large sets of signals
  //  by Dr.Ron Rubinstein , Israel Institute of Technology
  /// @param D 
  /// @param x 
  /// @param DtX 
  /// @param XtX 
  /// @param G 
  /// @param n 
  /// @param m 
  /// @param L 
  /// @param T 
  /// @param eps 
  /// @param gamma_mode 
  /// @param profile 
  /// @param msg_delta 
  /// @param erroromp 
  ///
  /// @return 
  ///
  void ompcore(double D[], double x[], double DtX[], double XtX[], double G[], unsigned n, unsigned m, unsigned L, int T, double eps, int gamma_mode, int profile, double msg_delta, int erroromp);
 
  /// 
  ///
  /// @param input 
  /// @param nchnl 
  /// @param psz 
  /// @param stepsz 
  /// @param patch2dMat 
  /// 
  void im2patchMat(MatrixXd const& input, unsigned nchnl, unsigned psz[2], unsigned stepsz[2], MatrixXd &patch2dMat);

  /// 
  ///
  /// @param file 
  /// @param rows 
  /// @param cols 
  ///
  void loadDct(const char* file,int rows, int cols, MatrixXd &);

  /// 
  ///
  /// @param mat 
  ///
  void remove_dc( MatrixXd &, char* );

  private:

  // Input: a vector
  // output: maxIdx 
  void maxIdxVec(VectorXd const& v, unsigned &maxIdx); 
  
  // compute w in L*w = G_{I,k} and update L
  // 
  inline void updateL(MatrixXd& L, MatrixXd const& G_Ik, vector<unsigned> const& I, unsigned k, bool &flag);

  // Input: low-triangular matrix L, rhs b type
  //        L*L^T *x = b(type "LL"),L*x = b(type "L")
  // Output: x
  inline void LL_solver(MatrixXd const& LL, VectorXd const& b, const char* type, VectorXd &x); 

};

}


#endif
