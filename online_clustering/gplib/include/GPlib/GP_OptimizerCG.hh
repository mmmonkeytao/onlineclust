#ifndef GP_OPTIMIZER_CG_HH
#define GP_OPTIMIZER_CG_HH

#include <gsl/gsl_multimin.h>
#include "GPlib/GP_ObjectiveFunction.hh"

namespace GPLIB {

  /*!
   * Opimizer class
   *
   * This class optimizes the GP objective function, i.e. it is 
   * the core of GP hyper parameter training
   */
  class GP_OptimizerCG
  {
    
  public:

    /*!
     * The constructor needs an objective function and some parameters
     */
    GP_OptimizerCG(GP_ObjectiveFunctionBase &fn, int length, double eps = 0.001);

    /*!
     * Initializes the optimizer
     */
    void Init(std::vector<double> const &init);
    
    /*!
     * Performs one iteration of the optimization. Returns true if improvement 
     * is still possible. Can be used to do:  while(m.Iterate()){}
     */
    bool Iterate();
    
    /*!
     * Returns the current arguments (= kernel params) of the objective function
     */
    void GetCurrValues(std::vector<double> &vals) const;

    /*!
     * Returns the current value of the objective function
     */
    double GetCurrentError() const;
    
    /*!
     * Checks whether the optimization has converged
     */
    bool  TestConvergence() const;

  private:

    const double _interv, _ext, _ratio, _sig, _rho;
    double _conv_thresh;
    uint _max, _iter, _n;
    int _length;
    bool _ls_failed;
    double _red, _x1, _x2, _x3, _x4, _f0, _f1, _f2, _f3, _f4, _d0, _d1, _d2, _d3, _d4;
    std::vector<double> _fX;
    GP_Vector _df0, _df3, _s, _X, _Z;
    std::string _method;

    GP_ObjectiveFunctionBase    &_objFn;

    void Extrapolation(uint &M, GP_Vector &X0, 
		       GP_Vector &dF0, double &F0);
    void Interpolation(uint &M, GP_Vector &X0, 
		       GP_Vector &dF0, double &F0);
		       
    bool NewSearchDirection(uint &M, GP_Vector const &X0, 
			    GP_Vector const &dF0, double F0);  

};


}



#endif
