#ifndef GP_REGRESSION_HH
#define GP_REGRESSION_HH

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

#include "GPlib/GP_Base.hh"
#include "GPlib/GP_OptimizerCG.hh"

namespace GPLIB {

  template<typename InputType, typename OutputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >
  class GP_Regression : public GP_Base<InputType, OutputType, KernelType>
  {
  public:

    typedef GP_Base<InputType, OutputType, KernelType> Super;
    typedef GP_Regression<InputType, OutputType, KernelType> Self;
    typedef typename Super::HyperParameters HyperParameters;


    GP_Regression() :
      GP_Base<InputType, GP_Vector, KernelType>(), 
      _ydim(0), _hparms(), _L(), _y(), _K_inv(),
      _alpha(), _logZ(0)
    {}

    GP_Regression(GP_DataSet<InputType, OutputType> const &train_data, 
		  HyperParameters const &hparms = HyperParameters()) :
      Super(train_data), 
      _ydim(train_data.GetOutputDim()),
      _hparms(hparms), _L(), _y(), _K_inv(), _alpha(), _logZ(0)
    {}


    virtual ~GP_Regression() 
    {}

    /*!
     * Retruns true if the class name is correctly given
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      std::string(classname) == "GP_Regression");
    }

    virtual void Estimation()
    {
      // first we compute the covariance matrix
      _L = Super::ComputeCovarianceMatrix(_hparms);

      //std::cout << "making Cholesky " << std::endl;
      if (!_L.Cholesky())
	throw GP_EXCEPTION("Cholesky decomposition of a non-positive definite matrix.");

      _y     = GP_Matrix(Super::GetY(), true);
      _alpha = _L.SolveChol(_y);

      ComputeLogZ();
      ComputeDerivLogZ();
    }


    virtual void UpdateModelParameters(HyperParameters const &new_hyp)
    {
      _hparms = new_hyp;
      _K_inv = GP_Matrix();
      Estimation();
    }


    /*!
     * Prediction for scalar output values and a single test input
     */
    void Prediction(InputType const &test_input,
		    double &mean, double &cov) const
    {
      uint n = Super::Size();
      GP_Vector k_star(n);
      for(uint i=0; i<n; i++)
	k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, _hparms);

      // first we compute the mean: mu = k_star^T * alpha
      mean = (k_star * _alpha)[0];

      // Solve L * x = k_star and compute variance
      GP_Vector v = _L.ForwSubst(k_star);
      cov = Super::CovFunc(test_input, test_input, _hparms) - v.Dot(v);
    }
  
    /*!
     * Prediction for vector output values and a single test input
    */
    void Prediction(InputType const &test_input,
		    GP_Vector &mean, GP_Matrix &cov) const
    {
      uint outdim = Super::GetOutputDim();
      uint n = Super::Size();
      GP_Vector k_star(n);
      for(uint i=0; i<n; i++)
	k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, _hparms);

      // first we compute the mean: mu = k_star^T * alpha
      mean = k_star * _alpha;

      double k_star_star = Super::CovFunc(test_input, test_input, _hparms);
    
      // Solve L * x = k_star and store in v
      GP_Vector v = _L.ForwSubst(k_star);
      double var = k_star_star - v.Dot(v);

      cov = GP_Matrix::Diag(GP_Vector(outdim, var));
    }
  
    void Prediction(GP_DataSet<InputType, OutputType> const &test_data, 
		    GP_Vector &mean, GP_Matrix &cov) const
    {
      GP_Matrix K_star = ComputeCovarianceMatrix(test_data, _hparms);
      GP_Matrix K_star_star = ComputeCovarianceMatrix(test_data, test_data, _hparms);

      GP_Matrix V = _L.ForwSubst(K_star);

      mean = K_star.Transp() * _alpha;
      cov = K_star_star - V.Transp() * V;
    }
  
    

    /*!
     * Returns the log-marginal
     */
    double GetLogZ() const
    {
      return _logZ;
    }

    /*!
     * Returns the derivative of the log-marginal with respect to the
     * kernel parameters
     */
    GP_Vector GetDeriv() const
    {
      return _deriv;
    }
    
    virtual double 
    LearnHyperParameters(std::vector<double> &init, 
			 GP_Vector lower_bounds = GP_Vector(1,0.0), 
			 GP_Vector upper_bounds = GP_Vector(1,1.0), 
			 uint nb_iterations = 0)
    {
      double residual;
      GP_Vector all_lbounds = _hparms.MakeBounds(lower_bounds);
      GP_Vector all_ubounds = _hparms.MakeBounds(upper_bounds);
      Optimization(init, all_lbounds, all_ubounds, residual);
      init = _hparms.ToVector();

      return residual;
    }


  protected:

    uint _ydim;
    HyperParameters _hparms;
    GP_Matrix _L, _y, _K_inv;
    GP_Matrix _alpha;
    double _logZ;     // log marginal likelihood
    GP_Vector _deriv;

  private:

    typedef GP_ObjectiveFunction<InputType, Self> ObjectiveFunction;

    void ComputeLogZ()
    {
      uint n = Super::Size();

      // compute y^T * alpha
      double sum = _y.ElemMult(_alpha).Sum();

      _logZ = -0.5 * sum - _L.LogTrace() - n/2. * LOG2PI;
    }

    
    void ComputeDerivLogZ()
    {
      uint n = Super::Size();

      // Compute inverse covariance matrix only if necessary
      if(_K_inv.Rows() != n || _K_inv.Cols() != n){
	_K_inv = _L.ForwSubst(GP_Matrix::Identity(n));
	_K_inv = _K_inv.TranspTimes(_K_inv);
      }

      GP_Matrix C, Z = _alpha.TimesTransp(_alpha) - _K_inv;

      _deriv = GP_Vector(_hparms.Size());
      for(uint j=0; j<_deriv.Size(); ++j){
	C = Super::ComputePartialDerivMatrix(_hparms, j);
	_deriv[j] = Z.ElemMult(C).Sum() / 2.;
      }
    }


    bool Optimization(std::vector<double> const &init_params, 
		      GP_Vector const &lower_bounds,
		      GP_Vector const &upper_bounds,
		      double &residual)
    {
      std::cout << "init " << std::flush;
      
      for(uint i=0; i<init_params.size(); ++i)
	std::cout << init_params[i] << " " << std::flush; 
      std::cout << std::endl;

      ObjectiveFunction err(*this, _hparms.Size(), lower_bounds, upper_bounds);
      GP_OptimizerCG min(err, -60);

      uint nb_params = init_params.size();
      uint data_dim = Super::GetTrainData().GetInputDim();
      std::vector<double> init(nb_params), cur_vals(nb_params);

      HyperParameters hparams(_hparms);
      hparams.FromVector(init_params);
      hparams.TransformInv(lower_bounds, upper_bounds);

      min.Init(hparams.ToVector());

      std::cout << "starting optimizer..." << std::endl;

      uint nb_max_iter = 50, iter = 0;
      _deriv = GP_Vector(_deriv.Size(), 1.);

      while(min.Iterate() && _deriv.Abs().Max() > 1e-5 && iter < nb_max_iter){

	min.GetCurrValues(cur_vals);
	
	HyperParameters cur_parms(cur_vals, data_dim);
	cur_parms.Transform(lower_bounds, upper_bounds);

	std::cout << "x " << std::flush;
	for(uint i=0; i<cur_parms.Size(); ++i)
	  std::cout << cur_parms.ToVector()[i] << " " << std::flush;
	std::cout << std::endl;
	std::cout << "deriv " << _deriv << std::endl;
	std::cout << "val " << min.GetCurrentError() << std::endl;

	++iter;
      }

      bool retval = min.TestConvergence();
      std::cout << std::endl;
      if(retval)
	std::cout << "converged!" << std::endl;

      min.GetCurrValues(init);
      _hparms.FromVector(init);
      _hparms.Transform(lower_bounds, upper_bounds);
      
      std::cout << "found parameters: " << std::flush;	
      for(uint i=0; i<_hparms.Size(); ++i){
	std::cout << _hparms[i] << " " << std::flush;
      }
      std::cout << std::endl;

      residual = min.GetCurrentError();      

      return retval;
    }


  };

}
#endif
