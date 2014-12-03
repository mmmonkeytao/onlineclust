#ifndef GP_REGRESSION_IVM_HH
#define GP_REGRESSION_IVM_HH

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

#include "GPlib/GP_Regression.hh"

namespace GPLIB {

  template<typename InputType, typename OutputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >
  class GP_RegressionIVM : public GP_Regression<InputType, OutputType, KernelType>
  {
  public:

    typedef GP_Regression<InputType, OutputType, KernelType> Super;
    typedef GP_RegressionIVM<InputType, OutputType, KernelType> Self;
    typedef typename Super::HyperParameters HyperParameters;


    GP_RegressionIVM() :
      GP_Regression<InputType, GP_Vector, KernelType>()
    {}

    GP_RegressionIVM(GP_DataSet<InputType, OutputType> const &train_data, 
		     uint activeSetSize,
		     HyperParameters const &hparms = HyperParameters(), 
		     double lambda = 1.0) :
      Super(train_data, hparms), _d(activeSetSize)
    {}


    virtual ~GP_RegressionIVM() 
    {}

    /*!
     * Retruns true if the class name is correctly given
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      std::string(classname) == "GP_RegressionIVM");
    }

    virtual void Estimation()
    {
      // Number of points in training data
      uint n = Super::Size();

      // Initialize site parameters
      GP_Vector m(n);
      GP_Vector beta(n);

      // Initialize approximate posterior parameters
      _mu   = GP_Vector(n);
      _zeta = GP_Vector(n);
      for(uint i=0; i<n; ++i)
	_zeta[i] = Super::CovFunc(Super::GetX()[i], Super::_hparms);

      std::cout << "zeta " << _zeta << std::endl;

      // Initialize active and passive set
      _I.clear();
      std::set<uint> J;
      for(uint i=0; i<n; ++i)
	J.insert(i);

      _M = GP_Matrix();
      _g = _nu = _delta = GP_Vector(_d);

      for(uint k=0; k<_d; ++k){

	// find next point for inclusion into the active set
	uint argmax = FindMostInformativePoint(J, beta, _mu, _zeta, 
					       _g[k], _nu[k], _delta[k], k);

	// refine site params, posterior params and matrices M, L, and K
	UpdateAll(argmax, _g[k], _nu[k], k, m, beta, _mu, _zeta, _M);
      
	// add idx to I and remove it from J
	_I.push_back(argmax);
	J.erase(argmax);
      }

      _mu_sqz.Resize(_d); // squeezed version of mu

      uint i=0;
      for(std::list<uint>::const_iterator it = _I.begin(); 
	  it != _I.end(); ++it, ++i)
	_mu_sqz[i] = _mu[*it];

      ComputeLogZ();
      ComputeDerivLogZ();
    }


    /*!
     * Prediction for scalar output values and a single test input
     */
    /*void Prediction(InputType const &test_input,
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
      }*/
  
    /*!
     * Prediction for vector output values and a single test input
    */
    /*void Prediction(InputType const &test_input,
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
    */
    

    virtual double 
    LearnHyperParameters(std::vector<double> &init, 
			 GP_Vector lower_bounds = GP_Vector(1,0.0), 
			 GP_Vector upper_bounds = GP_Vector(1,1.0), 
			 uint nb_iterations = 0)
    {
      return 0;
    }


  protected:

    std::list<uint> _I;
    uint _d, _old_d;
    GP_Vector _mu, _zeta, _g, _nu, _delta, _muL0, _muL1, _mu_part, _mu_sqz;
    GP_Matrix _M, _L2, _L3;

  private:

    typedef GP_ObjectiveFunction<InputType, Self> ObjectiveFunction;

    void ComputeLogZ()
    {
      uint n = Super::Size();

      if(_old_d == 0)
	Super::_logZ = 0;

      for(uint i=_old_d; i<_d; ++i)
      	Super::_logZ -= log(Super::_L[i][i]);

      if(_old_d == 0){
	_muL0 = Super::_L.ForwSubst(_mu);
	_muL1 = Super::_L.BackwSubst(_muL0);
	Super::_logZ -= _mu.Dot(_muL1) / 2.;
      }

      else {
	
	// use the Schur complement for efficiency
	GP_Vector mu_tilde = _mu.SubVector(_old_d, _d);

	GP_Vector tmp = _L3.ForwSubst(mu_tilde - _L2 * _muL0);

	_muL0.Append(tmp);

	Super::_logZ -= tmp.Dot(tmp) / 2.;
      }

      Super::_logZ -= n/2. * LOG2PI;
    }

        /*!
     * Computes the partial derivative of the log-posterior with respect to
     * the  posterior covariance  matrix (K + B^-1).  The function uses the 
     * covariance matrix and the mean value found after point selection,i.e.
     * the true posterior is approximated with the active set.
     */
    GP_Matrix ComputePartDerivCov() const
    {
      if(_muL1.Size() == 0)
	throw GP_EXCEPTION("muL not omputed");

      GP_Matrix I = GP_Matrix::Identity(_d);
      if(_old_d == 0)
	return (GP_Matrix::OutProd(_muL1) - Super::_L.SolveChol(I)) / 2.; 

      else {
	GP_Vector Cinv_m = Super::_L.SolveChol(_mu_sqz);
	
	return (GP_Matrix::OutProd(Cinv_m) - Super::_L.SolveChol(I)) / 2.; 
      }
    }

    /*!
     * Computes the derivative with respect to the kernel parameters
     */
    void ComputeDerivLogZ()
    {
      Super::_deriv = GP_Vector(Super::_hparms.Size());
      GP_Matrix Z2 = ComputePartDerivCov();

      for(uint k=0; k<Super::_deriv.Size(); ++k){
	/*
	 * CAREFUL HERE! This is a slight hack that only works with kernels where the
	 * last parameter is the data noise. 
	 */
	if(k == Super::_deriv.Size() - 1)
	  Super::_deriv[k] = Z2.Trace();
	else {
	  GP_Matrix C = Super::ComputePartialDerivMatrix(_I, Super::_hparms, k);
	  Super::_deriv[k] = Z2.ElemMult(C).Sum();	
	}
      }
    }


    template<typename InactiveSetContainer>
    uint FindMostInformativePoint(InactiveSetContainer const &J,
				  GP_Vector const &beta,
				  GP_Vector const &mu, GP_Vector const &zeta, 
				  double &g_kmax, double &nu_kmax, double &Delta_max, 
				  uint k, uint last = 0)
    {
      typename InactiveSetContainer::iterator argmax = J.begin();
      Delta_max = -HUGE_VAL;

      // loop over the inactive set and see what's interesting there
      for(typename InactiveSetContainer::iterator it_n = J.begin(); 
	  it_n != J.end(); ++it_n){
	
	uint idx = *it_n - last;

	// compute gradient g_kn and nu_kn
	double denom = zeta[idx] + 1./beta[idx];
	double g_kn  = (Super::GetY()[*it_n] - mu[idx]) / denom;
	double nu_kn = 1. / denom;
	
	// compute differential entropy score
	// the following line is the original code:
	double DeltaH_kn = -::log(1.0 - nu_kn * zeta[idx]) / (2. * LOG2);

	// however, as we are only interested in the maximum, we can also use this
	// here (and save some log computations):
	//double DeltaH_kn = nu_kn * zeta[idx];

	// update maximum and argmax
	if(DeltaH_kn > Delta_max){
	  Delta_max = DeltaH_kn;
	  g_kmax  = g_kn;
	  nu_kmax = nu_kn;
	  argmax  = it_n;
	}
      }

      return *argmax;
    }

    void UpdateAll(uint idx, double g, double nu, double k,
		   GP_Vector &m, GP_Vector &beta,
		   GP_Vector &mu, GP_Vector &zeta,
		   GP_Matrix &M, uint start = 0) 
    {
      uint idx2 = idx - start;

      // update m and beta
      if(fabs(g) < 1e-15)
	m[idx2] = mu[idx];
      else if(fabs(nu) > 1e-15)
	m[idx2]    = g / nu + mu[idx];
      else {
	throw GP_EXCEPTION("nu is zero and g is not zero!");
      }

      beta[idx2] = nu / ( 1. - nu * zeta[idx]);
      if(beta[idx2] < 1e-15)
	beta[idx2] = 1e-15;

      // compute zeta and mu
      GP_Vector s_nk, a_nk;
      UpdateMuZeta(k, idx, M, g, nu, s_nk, mu, zeta);

      // update M, L, and K
      UpdateMLK(k, idx, nu, s_nk, M);
    }

    void UpdateMuZeta(uint k, uint idx, GP_Matrix const &M, 
		      double g_kn, double nu_kn,
		      GP_Vector &s_nk, GP_Vector &mu, 
		      GP_Vector &zeta, uint last = 0) const
    {
      uint n = Super::Size();
      GP_Vector k_nk(n - last);

      for(uint i=last; i<n; ++i)
	k_nk[i - last] = Super::CovFunc(Super::GetX()[i], Super::GetX()[idx], 
					Super::_hparms);	  
      if(k == 0){
	s_nk = k_nk;
      }
      else {
	GP_Vector colvec = _M.Col(idx);
	if(colvec.Size() >= k)
	  colvec.Resize(k);
	s_nk = k_nk - colvec * M;
      }

      zeta -= nu_kn * s_nk.Sqr();
      mu   += g_kn * s_nk;
    }

    void UpdateMLK(uint k, uint idx, double nu_kn, 
		   GP_Vector const &s_nk, GP_Matrix &M)
    {
      UpdateLK(k, idx, nu_kn, s_nk, M);
      UpdateM(k, nu_kn, s_nk, M);
    }

    void UpdateM(uint k, double nu_kn, 
		 GP_Vector const &s_nk, GP_Matrix &M) const
    {
      // append sqrt(nu_kmax)s_nk to M_k-1
      double sqrt_nu_kn = sqrt(nu_kn);
      if(k == 0){
	M = GP_Matrix(1, s_nk.Size());
	for(uint i=0; i<s_nk.Size(); ++i)
	  M[0][i] = sqrt_nu_kn * s_nk[i];
      }
      else {
	M.AppendRow(sqrt_nu_kn * s_nk);
      }
    }

    void UpdateLK(uint k, uint idx, double nu_kn, 
		  GP_Vector const &s_nk, GP_Matrix const &M) 
    {
      double sqrt_nu_kn = sqrt(nu_kn);
      GP_Vector a_nk;
      if(idx < M.Cols())
	a_nk = M.Col(idx);	

      // update L
      if(k == 0){
	Super::_L = GP_Matrix(1, 1);
	Super::_L[0][0] = 1./sqrt_nu_kn;
      }

      else {	  
	// update L
	GP_Vector extra_col_L = GP_Vector(k+1);
	extra_col_L[k] = 1./sqrt_nu_kn;
	Super::_L.AppendRow(a_nk);
	Super::_L.AppendColumn(extra_col_L);
      }
    }

  };

}
#endif
