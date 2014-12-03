
#ifndef GP_SPARSE_REGRESSION_HH
#define GP_SPARSE_REGRESSION_HH

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include "GPlib/GP_Regression.hh"

namespace GPLIB {

  template<typename InputType, typename OutputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >
  class GP_SparseRegression : public GP_Regression<InputType, OutputType, KernelType>
  {
  public:

    typedef GP_Regression<InputType, OutputType, KernelType> Super;
    typedef typename Super::HyperParameters HyperParameters;


    GP_SparseRegression() :
      Super()
    {}

    GP_SparseRegression(GP_DataSet<InputType, OutputType> const &train_data, 
			uint activeSetSize, 
			HyperParameters const &hparms = HyperParameters()) :
      Super(train_data), _d(activeSetSize)
    {}


    virtual ~GP_SparseRegression() 
    {}

    /*!
     * Retruns true if the class name is correctly given
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      std::string(classname) == "GP_SparseRegression");
    }

    virtual void Estimation()
    {
      uint n = Super::Size();

      // Initialize site parameters
      GP_Vector m(n);
      GP_Vector beta(n);

      // Initialize approximate posterior parameters
      _mu   = GP_Vector(n);
      _zeta = GP_Vector(n);
      for(uint i=0; i<n; ++i)
	_zeta[i] = Super::CovFunc(Super::GetX()[i], Super::GetX()[i], 
				  Super::_hparms);

      // Initialize active and passive set
      _I.clear();
      std::set<uint> J;
      for(uint i=0; i<n; ++i)
	J.insert(i);

      _M = GP_Matrix();
      _g = _nu = GP_Vector(_d);

      for(uint k=0; k<_d; ++k){

	std::cout << "k = " << k << std::endl;

	// find next point for inclusion into the active set
	uint argmax = FindMostInformativePoint(J, _mu, _zeta, beta, _g[k], _nu[k]);

	std::cout << "updating " << std::endl;

	// refine site params, posterior params and matrices M, L, and K
	UpdateAll(argmax, _g[k], _nu[k], k, m, beta, _mu, _zeta, _M);
      
	// add idx to I and remove it from J
	_I.push_back(argmax);
	J.erase(argmax);
      }


      //std::vector<OutputType> y_sparse;
      std::vector<double> y_sparse;
      for(std::list<uint>::const_iterator it = _I.begin(); it != _I.end(); ++it)
	y_sparse.push_back(Super::GetY()[*it][0]);
      GP_Matrix y(y_sparse, false);
      Super::_alpha = Super::_L.SolveChol(y);

      std::cout << "alpha " << Super::_alpha.Rows() << " " << Super::_alpha.Cols() << std::endl;

      // re-compute site parameters for numerical stability
      //ComputeSiteParams(m, beta, _mu);

      // compute log posterior and  derivative wrt the kernel params
      //ComputeLogZ();
      //ComputeDerivLogZ();
 

      /*
      // first we compute the covariance matrix
      _L = Super::ComputeCovarianceMatrix(_hparms);

      std::cout << "making Cholesky " << std::endl;
      if (!_L.Cholesky())
	throw GP_EXCEPTION("Cholesky decomposition of a non-positive definite matrix.");

      std::cout << "computing alpha" << std::endl;
      GP_Matrix y(Super::GetY(), false);

      std::cout << "L size " << _L.Rows() << " " << _L.Cols() << std::endl;
      std::cout << "y size " << y.Rows() << " " << y.Cols() << std::endl;

      _alpha = _L.SolveChol(y);

      std::cout << "computing y^T * alpha" << std::endl;
      // compute y^T * alpha
      double sum = 0;
      for(uint i=0; i<y.Rows(); ++i)
	sum += y.Row(i).Dot(_alpha.Row(i));

      std::cout << "computing logZ " << std::endl;
      _logZ = -0.5 * sum - _L.Trace() - n/2. * log(2.0 * M_PI);*/
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
	k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, Super::_hparms);

      // first we compute the mean: mu = k_star^T * alpha
      mean = (k_star * Super::_alpha)[0];

      // Solve L * x = k_star and compute variance
      GP_Vector v = Super::_L.ForwSubst(k_star);
      cov = Super::CovFunc(test_input, test_input, Super::_hparms) - v.Dot(v);
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
	k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, Super::_hparms);

      // first we compute the mean: mu = k_star^T * alpha
      mean = k_star * Super::_alpha;

      double k_star_star = Super::CovFunc(test_input, test_input, Super::_hparms);
    
      // Solve L * x = k_star and store in v
      GP_Vector v = Super::_L.ForwSubst(k_star);
      double var = k_star_star - v.Dot(v);

      cov = GP_Matrix::Diag(GP_Vector(outdim, var));
    }
  
    void Prediction(GP_DataSet<InputType, OutputType> const &test_data, 
		    GP_Vector &mean, GP_Matrix &cov) const
    {
      GP_Matrix K_star = ComputeCovarianceMatrix(test_data, Super::_hparms);
      GP_Matrix K_star_star = ComputeCovarianceMatrix(test_data, test_data, Super::_hparms);

      GP_Matrix V = Super::_L.ForwSubst(K_star);

      mean = K_star.Transp() * Super::_alpha;
      cov = K_star_star - V.Transp() * V;
    }
    

  private:

    std::list<uint> _I;
    uint _d;
    GP_Vector _mu, _zeta, _g, _nu;
    GP_Matrix _M;

    void ComputeLogZ()
    {

    }

    
    void ComputeDerivLogZ()
    {

    }

    /*!
     * Numerically stable version to compute dlZ and d2lZ
     */
    void ComputeDerivatives(double y, double mu, double zeta, double beta,
			    double &g_in, double &nu_in) const
    {
      if(beta < 1e-15)
	g_in = 0;

      g_in  = (y - mu) / (1./beta + zeta);
      nu_in = 1./(zeta + 1./beta);
    }

    template<typename InactiveSetContainer>
    uint FindMostInformativePoint(InactiveSetContainer const &J,
				  GP_Vector const &mu, GP_Vector const &zeta, 
				  GP_Vector const &beta, 
				  double &g_kmax, double &nu_kmax)
    {
      typename InactiveSetContainer::iterator argmax = J.begin();
      double Delta_max = -HUGE_VAL;

      // loop over the inactive set and see what's interesting there
      for(typename InactiveSetContainer::iterator it_n = J.begin(); 
	  it_n != J.end(); ++it_n){
	
	// compute gradient g_kn and nu_kn
	double g_kn, nu_kn;
	//ComputeDerivatives(Super::GetY()[*it_n], mu[*it_n],
	//		   zeta[*it_n], beta[*it_n], g_kn, nu_kn);

	// Careful here! This is a hack! This needs to ne fixed!
	ComputeDerivatives(Super::GetY()[*it_n][0], mu[*it_n],
			   zeta[*it_n], beta[*it_n], g_kn, nu_kn);
	
	// compute differential entropy score
	// the following line is the original code:
	double DeltaH_kn = -::log(1.0 - nu_kn * zeta[*it_n]) / (2. * LOG2);
	// however, as we are only interested in the maximum, we can also use this
	// here (and save some log computations):
	//double DeltaH_kn = nu_kn * zeta[*it_n];

	// update maximum and argmax
	// use the next line for debugging (it makes sure the data is treated
	// in the same order as is done in standard EP, i.e. for comparison) 
	//if(*n == k){
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
		   GP_Matrix &M) 
    {
      // update m and beta
      m[idx]    = g / nu + mu[idx];
      beta[idx] = nu / ( 1. - nu * zeta[idx]);
      
      // compute zeta and mu
      GP_Vector s_nk, a_nk;
      UpdateMuZeta(k, idx, M, g, nu, s_nk, mu, zeta);

      // update M, L, and K
      UpdateMLK(k, idx, nu, s_nk, M);
    }


    void UpdateMuZeta(uint k, uint idx, GP_Matrix const &M, 
		      double g_kn, double nu_kn,
		      GP_Vector &s_nk, GP_Vector &mu, GP_Vector &zeta) const
    {
      uint n = Super::Size();
      GP_Vector k_nk(n);

      for(uint i=0; i<n; ++i)
	k_nk[i] = Super::CovFunc(Super::GetX()[i], Super::GetX()[idx], 
				 Super::_hparms);	  
      if(k == 0){
	s_nk = k_nk;
      }
      else {
	s_nk = k_nk - M.Col(idx) * M;
      }
      
      zeta = zeta - nu_kn * s_nk.Sqr();
      mu   = mu + g_kn * s_nk;
    }

    void UpdateMLK(uint k, uint idx, double nu_kn, 
		   GP_Vector const &s_nk, GP_Matrix &M)
    {
      UpdateLK(k, idx, nu_kn, s_nk, M);
      UpdateM(k, idx, nu_kn, s_nk, M);
    }

    void UpdateM(uint k, uint idx, double nu_kn, 
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

      // update L and K
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
