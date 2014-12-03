#ifndef HMP_H
#define HMP_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace onlineclust{

  using uchar = unsigned char;

  class HMP{

  public:  
    void hmp_core(MatrixXd &x);
    /// 
    ///
    /// @param D 
    /// @param X 
    /// @param splevel 
    /// @param SPcodes 
    ///
    void MaxPool_layer1_BOMP(MatrixXd const&D, MatrixXd const&X, uint splevel, MatrixXd &SPcodes);

    void MaxPool_layer2_BOMP();
    
    void mat2patch(MatrixXd const& im, const char*type, MatrixXd &patchMat);


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

    uint sparse_level;
        
  };

}
#endif

