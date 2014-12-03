#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_Matrix.hh"
#include "GPlib/GP_DataSet.hh"
#include "GPlib/GP_Tensor.hh"

using namespace std;
using namespace GPLIB;



BEGIN_PROGRAM(argc, argv)
{
  GP_Matrix mat(14, 12);

  uint k = 0;
  for(uint i=0; i<mat.Rows(); ++i)
    for(uint j=0; j<mat.Cols(); ++j)
      mat[i][j] = k++;
      

  std::cout << mat << std::endl;
  std::cout << mat.Col(10) << std::endl;

  GP_Matrix mat3(14, 5);
  for(uint i=0; i<mat3.Rows(); ++i)
    for(uint j=0; j<mat3.Cols(); ++j)
      mat3[i][j] = k++;

  GP_Matrix mat2 = mat.RemoveRowAndColumn(10);

  std::cout << mat2 << std::endl;

  std::cout << mat3 << std::endl;


  mat.AppendVert(mat3);

  GP_Matrix sm = mat.SubMatrix(3, 5, 6, 8);

  std::cout << mat << std::endl;
  std::cout << sm << std::endl;

  GP_Tensor ten(12, 4, 7);

  std::cout << ten(3, all(), 2) << std::endl;
  std::cout << ten.Col(3) << std::endl;
  std::cout << ten.Layer(3) << std::endl;

}END_PROGRAM
