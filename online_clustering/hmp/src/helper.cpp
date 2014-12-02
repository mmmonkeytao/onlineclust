#include "omp.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;
using namespace onlineclust;

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


void OMP::maxIdxVec(VectorXd const& v, unsigned &maxIdx)
{
  double max = -1;
  
  for(int i = 0; i < v.size(); i++){
    if(fabs(v.coeff(i)) > max){
      max = fabs(v.coeff(i));
      maxIdx = i;
    }
  }
}


void OMP::remove_dc(MatrixXd &X, const char* type)
{
  if(!strcmp(type, "column")){
    unsigned szr = X.rows();
    VectorXd mean = MatrixXd::Zero(1,X.cols());
    for(int i = 0; i < X.rows(); ++i){
      mean += X.row(i);
    }
    mean /= (double)szr;

    for(int i = 0; i < X.rows(); ++i){
      X.row(i) -= mean;
    }
  } else {
    cerr << "\nUnknown type in OMP::remove_dc.\n";
  }
}

