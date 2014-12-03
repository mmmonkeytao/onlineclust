#ifndef GP_BINARY_CLASSIFICATION_MI_HH
#define GP_BINARY_CLASSIFICATION_MI_HH

#include <list>
#include <set>
#include <limits>

#include "GPlib/GP_BinaryClassificationIVM.hh"

namespace GPLIB {

  template<typename InputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >

  /*!
   * \class GP_BinaryClassificationMI
   * 
   * Implementation of the Informative Vector Machine, a sparse version of 
   * a binary GP classifier
   */
  class GP_BinaryClassificationMI : 
    public GP_BinaryClassificationIVM<InputType, KernelType>
  {
  public:

    typedef GP_BinaryClassificationIVM<InputType, KernelType> Super;
    typedef typename Super::DataSet DataSet;
    typedef typename Super::HyperParameters HyperParameters;

    /*!
     * Default constructor
     */
    GP_BinaryClassificationMI() :
      Super()
    {}

    /*!
     * The constructor needs the training data, a number of active points,
     * the slope 'lambda' of the sigmoid, a flag 'runEP' that turns ADF into EP,
     * and the kernel hyper parameters
     */
    GP_BinaryClassificationMI(DataSet const &train_data, uint activeSetSize,
			       double lambda = 1.0, bool runEP = true,
			      std::vector<double> const &hparms = std::vector<double>()) :
      Super(train_data, activeSetSize, hparms, lambda)
    {
      for(uint i=0; i<Super::Size(); ++i)
	_J.push_back(i);
    }

    /*!
     * Checks whether the classname is the correct one
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) || 
	      std::string(classname) == "GP_BinaryClassificationMI");
    }

    /*!
     * Computes the active set I, and all other required values 
     * (see base class version of Estimation()).
     */
    virtual void Estimation()
    {
      std::cout << "estimation, kernel params: " 
		<< Super::_hparms[0] << " "
		<< Super::_hparms[1] << " "
		<< Super::_hparms[2] << " " << std::endl;

      // Number of points in training data
      uint n = Super::Size();

      // Initialize site parameters
      GP_Vector m(n);
      GP_Vector beta(n);

      // Initialize approximate posterior parameters
      Super::_mu   = GP_Vector(n);
      Super::_zeta = GP_Vector(n);
      for(uint i=0; i<n; ++i)
	Super::_zeta[i] = Super::CovFunc(Super::GetX()[i], Super::GetX()[i], 
					 Super::_hparms);

      // Initialize active and passive set
      Super::_I.clear();
      _J.clear();
      for(uint i=0; i<n; ++i)
	_J.push_back(i);

      Super::_M = GP_Matrix();
      Super::_g = Super::_nu = Super::_delta = GP_Vector(Super::_d);

      for(uint k=0; k<Super::_d; ++k){

	// find next point for inclusion into the active set
	//	std::list<uint>::iterator argmax = 
	uint argmax =
	  FindMaxMIPoint(Super::_mu, Super::_zeta, 
			 Super::_g[k], Super::_nu[k], Super::_delta[k]);

	// refine site params, posterior params and matrices M, L, and K
	Super::UpdateAll(argmax, Super::_g[k], Super::_nu[k], k, m, beta, 
			 Super::_mu, Super::_zeta, Super::_M);
      
	// add idx to I and remove it from J
       	Super::_I.push_back(argmax);
	//_J.erase(argmax);
      }

      // re-compute site parameters for numerical stability
      Super::ComputeSiteParams(m, beta, Super::_mu, Super::_I.begin());

      // compute log posterior and  derivative wrt the kernel params
      Super::ComputeLogZ();
      Super::ComputeDerivLogZ();
    }


  protected:
  
    std::list<uint> _J;
    GP_Matrix _K, _Kpsv, _Lpsv;

    //std::list<uint>::iterator 
    uint
    FindMaxMIPoint(GP_Vector const &mu, GP_Vector const &zeta, 
		   double &g_kmax, double &nu_kmax, double &MI_max)
    {

      std::cout << "=== FindMaxMIPoint ===" << std::endl;

      uint j = Super::_I.size();
      InputType const &x_star = Super::GetX()[j];
      double sigma_Y = Super::CovFunc(x_star, x_star, Super::_hparms);
      MI_max = -HUGE_VAL;
      std::list<uint>::iterator argmax = _J.begin();

      std::cout << "sigma_Y = " << sigma_Y << std::endl;
      std::cout << "next point to test: " << x_star << std::endl;

      // loop through all passive points and see which one we can remove
      for (std::list<uint>::iterator candidate = _J.begin(); 
	   candidate != _J.end(); candidate++) {

	// we remove this point and compute the MI, but 
	// we have to add the point later again
	std::cout << "J size " << _J.size() << std::endl;
	uint cand_idx = *candidate;
	std::list<uint>::iterator cand_pos = _J.erase(candidate);
	double enumer = sigma_Y;

	std::cout << "\tnext candidate " << cand_idx << std::endl;
	std::cout << "J size " << _J.size() << std::endl;

	// if we don't have an active set, there is nothing to do here
	if(j != 0){

	  std::cout << "got an active set already" << std::endl;

	  // Compute k_star
	  GP_Vector k_star_act(Super::_I.size());
	  uint i=0;
	
	  for(std::list<uint>::const_iterator n = Super::_I.begin(); 
	      n != Super::_I.end(); ++n, ++i)
	    k_star_act[i] = Super::CovFunc(Super::GetX()[*n], x_star, 
					   Super::_hparms);
	
	  // compute (B^-1 + K)^-1 * k_star using our Cholesky decomp 
	  GP_Vector v = Super::_L.SolveChol(k_star_act);
	  enumer = sigma_Y - k_star_act.Dot(v);
	}

	std::cout << "running estimation passive" << std::endl;

	// compute Sigma of passive set
	EstimationPassive(cand_idx);
	
	// compute square root of site variances
	Super::UpdateSqrtS();

	std::cout << "getting k_star passive" << std::endl;
	// Compute k_star passive
	GP_Vector k_star_psv(_J.size());
	uint i = 0;
	for(std::list<uint>::const_iterator n = _J.begin(); 
	    n != _J.end(); ++n, ++i) {
	  k_star_psv[i] = Super::CovFunc(Super::GetX()[*n], x_star, Super::_hparms);
	}
	
	std::cout << "computing posterior covariance passive" << std::endl;

	// compute posterior covariance of the passive set
	GP_Vector v = _Lpsv.ForwSubst(Super::_s_sqrt * k_star_psv);
	double denom = MAX(sigma_Y - v.Dot(v), 0);

	// insert the point again for further rounds
	cand_pos = _J.insert(cand_pos, cand_idx);

	// now we can compute the mutual information
	double mi = enumer / denom;

	std::cout << "got MI " << mi << std::endl;

	// and store the maximum
	if(mi > MI_max){
	  MI_max = mi;
	  argmax = cand_pos;

	  std::cout << "found new max " << *argmax << ", " << MI_max << std::endl;
	}
      }      

      std::cout << "argmax is " << *argmax << std::endl;

      Super::ComputeDerivatives(Super::GetY()[*argmax], mu[*argmax] / zeta[*argmax],
				1./zeta[*argmax], g_kmax, nu_kmax);
      std::cout << "got derivs " << g_kmax << " " << nu_kmax << std::endl;

      _K = _K.RemoveRowAndColumn(*argmax);

      uint retval = *argmax;
      _J.erase(argmax);
      return retval;
    }

    // computes Sigma of passive set
    void EstimationPassive(uint cand_idx)
    {
      // first we compute the covariance matrix
      if(_K.Rows() == 0)
	_K = Super::ComputeCovarianceMatrix(Super::_hparms);
      
      uint rem_idx = cand_idx;
       for(std::list<uint>::const_iterator it = Super::_I.begin();
	  it != Super::_I.end(); ++it){
	if(cand_idx >= *it)
	  --rem_idx;
      }

      _Kpsv = _K.RemoveRowAndColumn(rem_idx);

      // initialize the site and posterior params
      uint n = _J.size()-1;
      Super::_nu_tilde = GP_Vector(n);
      Super::_tau_tilde = GP_Vector(n);

      GP_Vector mu(n);
      GP_Matrix Sigma = _Kpsv;

      // now we run EP until convergence (or max number of iterations reached)
      uint iter = 0;
      double max_delta = 1.;
      do {

	// run EP; the maximum difference in tau is our convergence criterion
	GP_Vector delta_tau = Super::ExpectationPropagation(mu, Sigma);
	max_delta = delta_tau.Abs().Max();

	// re-compute mu and Sigma for numerical stability; also computes L
	ComputePosteriorParams(mu, Sigma);

      } while(max_delta > Super::_EPthresh && ++iter < Super::_maxEPiter);
      
      // Does not happen often, it's not a big problem even if it does. 
      // If it happens too often, just increase '_maxEPiter'
      if(iter == Super::_maxEPiter){
	std::cerr << "Warning! EP did not converge" << std::endl;
      }
    }

    /*!
     * Computes mu and Sigma from the site parameters nu_tilde and tau_tilde
     */
    void ComputePosteriorParams(GP_Vector &mu, GP_Matrix &Sigma)
    {
      uint n = mu.Size();
      Super::UpdateSqrtS();

      _Lpsv = _Kpsv;
      _Lpsv.CholeskyDecompB(Super::_s_sqrt);

      // Compute S^0.5 * K in place
      Sigma = GP_Matrix(n, n);
      for(uint i=0; i<n; i++)
	for(uint j=0; j<n; j++)
	  Sigma[i][j] = Super::_s_sqrt[i] * _Kpsv[i][j];
      
      GP_Matrix V = _Lpsv.ForwSubst(Sigma);

      Sigma = _Kpsv - V.TranspTimes(V);
      mu    = Sigma * Super::_nu_tilde;
    }


  };

}
#endif
