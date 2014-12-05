#ifndef HMP_H
#define HMP_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace onlineclust{

  using uchar = unsigned char;
  // first: width, second: height
  using matSize = pair<uint, uint>;
  //using Mat3D = Matrix<VectorXd, Dynamic, Dynamic >;
  
  class HMP{

  public:  
    void hmp_core(MatrixXd &x, const char *type, uint SPlevel[2], MatrixXd &fea);
    /// 
    ///
    /// @param D 
    /// @param X 
    /// @param splevel 
    /// @param SPcodes 
    ///
    void MaxPool_layer1_mode1(MatrixXd &Gamma, matSize const&patchsz, matSize const&realsz);

    void MaxPool_layer2(MatrixXd &fea, matSize const&feaSz, uint pool[3]);
    
    void mat2patch(MatrixXd const& im, const char*type, matSize const& rsz, MatrixXd &patchMat);

    void loadDct(const char* flayer1, const char*flayer2, const char* type);
    

    struct patchsz{
      const uint width = 5;
      const uint height = 5;
    }patchsz;
    
    struct stepsz{
      const uint width1 = 1;
      const uint height1 = 1;
      const uint width2 = 1;
      const uint height2 = 1;
    }stepsz;

  private:
    const double eps = 0.1;
    MatrixXd D1rgb;
    MatrixXd D2rgb;
    MatrixXd D1depth;
    MatrixXd D2depth;
    
  };

}
#endif

