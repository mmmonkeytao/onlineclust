#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "hmp.h"

using namespace onlineclust;
using namespace onlineclust::omp;
using namespace std;
using namespace std::chrono;
using namespace Eigen;

int main(){
  

  // {
  //   // test maxIdxVec function
  //   VectorXd a{6};
  //   a << 1,4,4,12,-2,4;

  //   unsigned result;
  //   maxIdxVec(a, result);
  //   cout << "vector:\n" << a.transpose() << endl
  // 	 << "max variable idx:\n" << result << endl; 
  // }

  // {
  //   // Cholesky solver for type L*L^T *x = b(type "LL"),L*x = b(type "L")
  //   // text LL_solver function with L already given from cholesky decomposition
  //   VectorXd a;
  //   MatrixXd D{16,5}; D.setRandom();
  //   MatrixXd A{5,5}; A = D.transpose() * D;
  //   cout << "\nMatrix A:\n" << A << endl;
  //   a = VectorXd{5}; 
  //   a.setRandom(); 
  //   cout << "\nb: " << a.transpose() << endl;
  //   LLT<MatrixXd> llt{A};
  //   MatrixXd L = llt.matrixL();
  //   VectorXd x{5};
  //   LL_solver(L, a, "LL", x);
  //   cout << "Matrix L is:\n" << L << endl
  //   	 << "\nLL solver result x:\n" << x.transpose() << endl
  //   	 << "Eigen solver result x:\n" << llt.solve(a).transpose() << endl;
  // }  

  //  {
  //   // test time consumption for two different operation on matrix in Eigen
  //   unsigned dim = 1000;
  //   MatrixXd m1{dim,dim};
  //   m1.setRandom(); 
  //   auto s1 = high_resolution_clock::now();
  //   m1.conservativeResize(dim+1,dim+1);
  //   m1.bottomRows(1) = MatrixXd::Ones(1,dim+1);
  //   m1.rightCols(1) = MatrixXd::Zero(dim+1,1);
  //   auto e1 = high_resolution_clock::now();
  //   cout << endl << "m1's duration of resize operation:(milliseconds) \n" 
  // 	 << duration_cast<microseconds>(e1-s1).count()<<endl;
  
  //   MatrixXd m2{dim+1,dim+1}, rand1{dim,dim}, rand3{dim,dim};
  //   m2.setRandom();  
  //   rand3.setRandom();
  //   s1 = high_resolution_clock::now();
  //   rand1 = rand3; 
  //   rand3 = MatrixXd{dim+1,dim+1};
  //   m2.topLeftCorner(dim,dim) = rand1;
  //   m2.bottomRows(1) = MatrixXd::Ones(1,dim+1);
  //   m2.rightCols(1) = MatrixXd::Zero(dim+1,1);
  //   rand3 = m2;
  //   e1 = high_resolution_clock::now();

  //   cout << "m2's duration of copy operation:(milliseconds) " 
  // 	 << duration_cast<microseconds>(e1-s1).count()<<endl;
  //   //################## Conclusion: copy takes less time than resize()#######
  // }

  {
    // test updateL function (to be done)
    // unsigned dim = 50;
    // MatrixXd L{dim, dim};
    // L = L.triangularView<Eigen::UnitUpper>();

    // MatrixXd G{1000,1000}; 
    // MatrixXd D{dim,1000}; D.setRandom();
    // G = D.transpose() * D;
    // vector<unsigned> I = {0,2};
    // unsigned k = 2;
    // updateL(L,G,I,k);
    // cout << "new L is:\n" << L << endl;
  }

  {
    // test Batch_OMP code
    MatrixXd X{15,1};
    X.setRandom();
    MatrixXd D{15,1000};
    D.setRandom();
    MatrixXd Gamma;
    auto t1 = high_resolution_clock::now();
    Batch_OMP(X,D,4,Gamma);
    auto t2 = high_resolution_clock::now();
    cout << "Sparse codes is:\n" << X-D*Gamma << endl
  	 << "computation time: (micros)" << duration_cast<milliseconds>(t2-t1).count() << endl;
 }
  return 0;
}
