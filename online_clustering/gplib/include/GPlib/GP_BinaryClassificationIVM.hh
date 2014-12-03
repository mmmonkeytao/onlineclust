#ifndef GP_BINARY_CLASSIFICATION_IVM_HH
#define GP_BINARY_CLASSIFICATION_IVM_HH

#include <list>
#include <set>
#include <limits>

#include "GPlib/GP_BinaryClassificationEP.hh"

namespace GPLIB {

  template<typename InputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >

  /*!
   * \class GP_BinaryClassificationIVM
   * 
   * Implementation of the Informative Vector Machine, a sparse version of 
   * a binary GP classifier
   */
  class GP_BinaryClassificationIVM : 
    public GP_BinaryClassificationEP<InputType, KernelType>
  {
  public:

    typedef GP_BinaryClassificationEP<InputType, KernelType> Super;
    typedef typename Super::DataSet DataSet;
    typedef typename Super::HyperParameters HyperParameters;

    /*!
     * Default constructor
     */
    GP_BinaryClassificationIVM() :
      Super(), _d(0), _old_d(0), _runEP(false)
    {}

    /*!
     * The constructor needs the training data, a number of active points,
     * the slope 'lambda' of the sigmoid, a flag 'runEP' that turns ADF into EP,
     * and the kernel hyper parameters
     */
    GP_BinaryClassificationIVM(DataSet const &train_data, uint activeSetSize,
			       std::vector<double> const &hparms = std::vector<double>(),
			       double lambda = 1.0, 
			       GP_Optimizer::OptimizerType type = GP_Optimizer::PR,
			       double step = 0.01, double tol = 1e-2, double eps = 1e-3,
			       bool runEP = true, bool verbose = true) :
      Super(train_data, hparms, lambda, type, step, tol, eps, verbose), _d(activeSetSize),
      _old_d(0), _runEP(runEP)
    { }

    /*!
     * Checks whether the classname is the correct one
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) || 
	      std::string(classname) == "GP_BinaryClassificationIVM");
    }

    GP_Matrix const &GetM() const
    {
      return _M;
    }

    /*!
     * Returns the mean of the latent posterior
     */
    virtual GP_Vector const &GetPosteriorMean() const
    {
      return _mu;
    }


    /*!
     * Computes mean and covariance of the posterior when adding a new
     * data point to the training set. The point is not actually added.
     */
    virtual void GetMuZeta(InputType const &input,
			   double &last_mu, double &last_zeta)
    {
      // Initialization
      last_mu = 0;
      last_zeta = Super::CovFunc(input, input, Super::_hparms);

      // we only need a new last column of M,
      // but we don't actually add it to M 
      GP_Vector lastMcol;

      uint k=0;
      for(std::list<uint>::const_iterator it = _I.begin(); 
	  it != _I.end(); ++it, ++k){
	
	// compute zeta and mu
	double s_nk_last, k_nk_last;
	k_nk_last = Super::CovFunc(input, Super::GetX()[*it], Super::_hparms);

	if(k == 0){
	  s_nk_last = k_nk_last;
	}
	
	else {
	  double dot = 0;
	  for(uint j=0; j<lastMcol.Size(); ++j)
	    dot += _M[j][*it] * lastMcol[j];
	  
	  s_nk_last = k_nk_last - dot;
	}
	
	last_mu   += _g[k] * s_nk_last;
	last_zeta -= _nu[k] * SQR(s_nk_last);

	// update last col of M by appending sqrt(nu_kmax)s_nk 
	lastMcol.Append(sqrt(_nu[k]) * s_nk_last);
      }
    }

    /*!
     * Computes the active set I, and all other required values 
     * (see base class version of Estimation()).
     */
    virtual void Estimation()
    {
      if(Super::_verbose)
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
      _mu   = GP_Vector(n);
      _zeta = GP_Vector(n);
      for(uint i=0; i<n; ++i)
	_zeta[i] = Super::CovFunc(Super::GetX()[i], Super::_hparms);

      // Initialize active and passive set
      _I.clear();
      std::set<uint> J;
      for(uint i=0; i<n; ++i)
	J.insert(i);

      _M = GP_Matrix();
      _g = _nu = _delta = GP_Vector(_d);

      for(uint k=0; k<_d; ++k){

	// find next point for inclusion into the active set
	uint argmax = FindMostInformativePoint(J, _mu, _zeta, 
					       _g[k], _nu[k], _delta[k], k);

	//std::cout << "delta[" << k << "] " << _delta[k] << std::endl;

	// refine site params, posterior params and matrices M, L, and K
	UpdateAll(argmax, _g[k], _nu[k], k, m, beta, _mu, _zeta, _M);
      
	// add idx to I and remove it from J
	_I.push_back(argmax);
	J.erase(argmax);
      }

      // re-compute site parameters for numerical stability
      ComputeSiteParams(m, beta, _mu, _I.begin());

      // compute log posterior and  derivative wrt the kernel params
      ComputeLogZ();
      ComputeDerivLogZ();
    }

    virtual void EstimationGivenI()
    {
      if(Super::_verbose)
	std::cout << "estimation given I, kernel params: " 
		  << Super::_hparms[0] << " "
		  << Super::_hparms[1] << " "
		  << Super::_hparms[2] << " " << std::endl;

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

      _M = GP_Matrix();
      _g = _nu = GP_Vector(_d);

      std::list<uint>::const_iterator it;
      uint k=0;
      for(it = _I.begin(); it != _I.end(); ++it, ++k){

	// find next point for inclusion into the active set
	Super::ComputeDerivatives(Super::GetY()[*it], _mu[*it] / _zeta[*it],
				  1./_zeta[*it], _g[k], _nu[k]);

	// refine site params, posterior params and matrices M, L, and K
	UpdateAll(*it, _g[k], _nu[k], k, m, beta, _mu, _zeta, _M);
      }

      // re-compute site parameters for numerical stability
      ComputeSiteParams(m, beta, _mu, _I.begin());

      // compute log posterior and  derivative wrt the kernel params
      ComputeLogZ();
      ComputeDerivLogZ();
    }

    void AddTrainingData(DataSet const &new_data)
    {
      Super::GetTrainData().Append(new_data);
      _old_d = _d;
    }

    uint GetNbNewActivePoints() const
    {
      return _d - _old_d;
    }

    virtual void EstimationIncremental(uint d_inc)
    {
      if(Super::_nu_tilde.Size() == 0 || Super::_tau_tilde.Size() == 0)
	throw GP_EXCEPTION("Could not do incremental estimation. "
			   "Run standard estimation first.");

      if(Super::_verbose)
	std::cout << "incremental estimation, kernel params: " 
		  << Super::_hparms[0] << " "
		  << Super::_hparms[1] << " "
		  << Super::_hparms[2] << " " << std::endl;

      // Number of points in training data
      uint n = Super::Size();
      uint n_old = _mu.Size();
      uint n_diff = n - n_old;
      if(n_diff == 0)
	return ; // no new training data -> nothing to be done

      // New active set size
      _d += d_inc;

      // Initialize site parameters
      GP_Vector m(n_diff);
      GP_Vector beta(n_diff);

      // Initialize approximate posterior parameters
      GP_Vector mu_inc(n_diff);
      GP_Vector zeta_inc(n_diff);
      for(uint i=n_old; i<n; ++i)
	zeta_inc[i-n_old] = Super::CovFunc(Super::GetX()[i], Super::_hparms);

      // Initialize passive set
      std::set<uint> J;
      for(uint i=n_old; i<n; ++i)
	J.insert(i);

      GP_Matrix M_inc;
      _g.Resize(_d);
      _nu.Resize(_d); 
      _delta.Resize(_d);

      // get the internals up to date (now we have more data)
      std::list<uint>::const_iterator it;
      uint k=0;
      GP_Vector s_nk_inc;

      for(it = _I.begin(); it != _I.end(); ++it, ++k){
	UpdateMuZeta(k, *it, M_inc, _g[k], _nu[k], s_nk_inc, mu_inc, zeta_inc, n_old);
	UpdateM(k, _nu[k], s_nk_inc, M_inc);
      }

      _mu.Append(mu_inc);
      _zeta.Append(zeta_inc);
      _M.AppendVert(M_inc);
      
      for(uint k=_old_d; k<_d; ++k){

	// find next point for inclusion into the active set
	uint argmax = FindMostInformativePoint(J, mu_inc, zeta_inc, 
					       _g[k], _nu[k], _delta[k], k, n_old);

	// refine site params, posterior params and matrices M, L, and K
	UpdateAll(argmax, _g[k], _nu[k], k, m, beta, _mu, _zeta, _M, n_old);
      
	mu_inc   = _mu.SubVector(n_old, n);
	zeta_inc = _zeta.SubVector(n_old, n);

	// add idx to I and remove it from J
	_I.push_back(argmax);
	J.erase(argmax);
	
	if(k == _old_d){
	  it = _I.end();
	  it--;
	}
      }

      // re-compute site parameters for numerical stability
      ComputeSiteParams(m, beta, _mu, it, n_old);

      if(_old_d != 0){
	_L2  = Super::_L.SubMatrix(0, _old_d, _old_d, _d);
	_L3  = Super::_L.SubMatrix(_old_d, _old_d, _d, _d);
      }

      // compute log posterior and  derivative wrt the kernel params
      ComputeLogZ();
      ComputeDerivLogZ();
    }


    /*!
     * Removes all non-active points from the training set. Careful here! 
     * This only makes sense after training and if training is not re-run
     */
    void Squeeze()
    {
      typename Super::DataSet new_train_data;

      std::list<uint> new_I;
      uint idx = 0;
      for(std::list<uint>::const_iterator it = _I.begin(); it != _I.end(); ++it, ++idx){
	new_train_data.Add(Super::GetX()[*it], Super::GetY()[*it]);
	new_I.push_back(idx);
      }
      
      _I = new_I;
      Super::GetTrainData() = new_train_data;
    }

    GP_Vector const &GetZeta() const
    {
      return _zeta;
    }


    GP_Vector const &GetActivePointDeltas() const
    {
      return _delta;
    }

    GP_Vector GetActivePointZetas() const
    {
      GP_Vector zeta(_I.size());

      std::list<uint>::const_iterator it = _I.begin();
      for(uint i=0; i<zeta.size(); ++i, ++it)
	zeta[i] = _zeta[*it];

     return zeta;
    }

    std::vector<uint> GetLeastInformativePoints(uint nb_points) const
    {
      std::vector<uint> li_points;
      std::list<std::pair<uint, double> >::const_iterator it = _point_entropies.begin();
      for(uint i=0; i<MIN(nb_points, _point_entropies.size()); ++i, ++it)
	    li_points.push_back(it->first);

      return li_points;
    }

    std::vector<uint> GetLeastVariantPoints(uint nb_points) const
    {
      std::vector<uint> lv_points;
      std::list<std::pair<uint, double> > zeta_idcs;

      for(uint i=0; i< _zeta.Size(); ++i)
	zeta_idcs.push_back(std::make_pair(i, _zeta[i]));

      zeta_idcs.sort(LessThan());

      std::list<std::pair<uint, double> >::const_iterator it = zeta_idcs.begin();
      for(uint i=0; i<MIN(nb_points, zeta_idcs.size()); ++i, ++it){
	lv_points.push_back(it->first);
      }

      return lv_points;
    }

    virtual void UpdateModelParameters(HyperParameters const &new_hyp)
    {
      Super::_hparms = new_hyp;

      
      if(_I.size() != 0){
	Super::_L = Super::ComputeCovarianceMatrix(_I, Super::_hparms);

	if(_runEP)
	  Super::_K = Super::_L;

	for(uint i=0; i<_I.size(); ++i)
	  Super::_L[i][i] += 1./Super::_tau_tilde[i];
	Super::_L.Cholesky();
      }

      ComputeLogZ();
      ComputeDerivLogZ();
      
      //std::cout << "L1 " << Super::_L << std::endl;
      //std::cout << "tt1 " << Super::_tau_tilde << std::endl;
      //std::cout << "deriv1 " << Super::GetDeriv() << std::endl;

      //EstimationGivenI();

      //std::cout << "L2 " << Super::_L << std::endl;
      //std::cout << "tt2 " << Super::_tau_tilde << std::endl;
      //std::cout << "logZ " << Super::_logZ << std::endl;
      //std::cout << "derivative " << Super::GetDeriv() << std::endl;
    }

    /*!
     * Learns hyper parameters for the kernel. In addition to the function provided
     * by the base class, this one uses the number of iterations. 
     */
    virtual double LearnHyperParameters(std::vector<double> &init, 
					GP_Vector lower_bounds = GP_Vector(1,0.), 
					GP_Vector upper_bounds = GP_Vector(1,0.), 
					uint nb_iterations = 5)
    {
      double residual = 0, min_res = std::numeric_limits<double>::max();
      HyperParameters argmin;
      
      GP_Vector all_lbounds = Super::_hparms.MakeBounds(lower_bounds);
      GP_Vector all_ubounds = Super::_hparms.MakeBounds(upper_bounds);
      
      // first we set the initial parameters
      Super::_hparms.FromVector(init);
      for(uint i=0; i<nb_iterations; ++i){

	Estimation();

	Super::_hparms.FromVector(init);
	Super::Optimization(all_lbounds, all_ubounds, residual);

	if(Super::_verbose)
	  std::cout << "optimized: " 
		    << Super::_hparms[0] << " "
		    << Super::_hparms[1] << " " 		  
		    << Super::_hparms[2] << " " << std::endl;

	if(residual < min_res){
	  min_res = residual;
	  argmin = Super::GetHyperParams();
	}
      }
      
      Super::_hparms = argmin;
      init = argmin.ToVector();
      Estimation();

      return min_res;
    }

    uint GetActiveSetSize() const
    {
      return _d;
    }

    std::list<uint> const &GetActiveSet() const
    {
      return _I;
    }


    /*!
     * Computes the log likelihood of the entire data set, not only the active points
     * Therefore, site paramters 'mu' and 'zeta' have to be vectors of size 'n' (not of size _I.size())
     */
    double GetLogLikelihood(GP_Vector const &mu, GP_Vector const &zeta, 
			    double bias) const
    {
      std::vector<int> y(Super::GetY());
      double ll = 0;
      for(uint i=0; i<y.size(); ++i)
	ll += Super::GetSigFunc().LogLikelihoodBin(y[i], mu[i], 
						   Super::_lambda, zeta[i], bias);

      return ll;
    }    

    /*!
     * Writes all active points into an ASCII file
     */
    void ExportActiveSet(char const *filename = "active_set") const
    {
      static uint idx = 0;

      std::stringstream fname;
      fname << filename << std::setw(3) << std::setfill('0') << idx++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());
      for(std::list<uint>::const_iterator it = _I.begin(); it != _I.end(); ++it){
	ofile << *it << " " << Super::GetX()[*it] << " " 
	      << Super::GetY()[*it] << std::endl;
      }
      ofile.close();
    }

    void ExportZeta(char const *filename = "zeta") const
    {
      static uint idx = 0;

      if(_zeta.Size() != Super::Size())
	std::cout << "Zeta is not as long as the training data! Could not export it." << std::endl;

      std::stringstream fname;
      fname << filename << std::setw(3) << std::setfill('0') << idx++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());
      for(uint i=0; i<_zeta.Size(); ++i){
	ofile << _zeta[i] << " "<< Super::GetY()[i] << std::endl;
      }
      ofile.close();
    }

    void ExportLeastInformative(char const *filename = "least_informative") const
    {
      static uint idx = 0;

      std::stringstream fname;
      fname << filename << std::setw(3) << std::setfill('0') << idx++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());
      std::list<std::pair<uint, double> >::const_iterator it = _point_entropies.begin();

      for(;it != _point_entropies.end(); ++it){	  
	ofile << it->first << " " << it->second << std::endl;
      }
      ofile.close();
    }

    double Prediction(InputType const &test_input) const
    {
      double mu_star, sigma_star;
      return Prediction(test_input, mu_star, sigma_star);
    }

    virtual double Prediction(InputType const &test_input, 
			      double &mu_star, double &sigma_star) const
    {
      if(Super::_nu_tilde.Size() == 0 || Super::_tau_tilde.Size() == 0)
	throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

      if(Super::_L.Rows() == 0 || Super::_L.Cols() == 0)
	throw GP_EXCEPTION("Cholesky matrix not computed. "
			   "Run 'Expectation Propagation' first.");
    
      // Compute k_star
      GP_Vector k_star(_I.size());
      uint i=0;
      for(std::list<uint>::const_iterator it = _I.begin(); it != _I.end(); ++it, ++i)
	k_star[i] = Super::CovFunc(Super::GetX()[*it], test_input, Super::_hparms);


      // compute (B^-1 + K)^-1 * k_star using our Cholesky decomp 
      GP_Vector v = Super::_L.SolveChol(k_star);

      // compute predictive mean and covariance
      GP_Vector mu_tilde = Super::GetSiteMean();
      uint n = _I.size();
      mu_star = (mu_tilde + Super::_bias * GP_Vector(n, 1.0)).Dot(v);
      double kk = Super::CovFunc(test_input, Super::_hparms);
      sigma_star = (kk - k_star.Dot(v));

      // the class probability we want is the expected value of the 
      // sigmoid under the predictive distribution
      double pi_star = 
	Super::SigFunc(mu_star / sqrt(1./SQR(Super::_lambda) + sigma_star));

      return pi_star;
    }
    
    virtual double PredictionIncremental(InputType const &test_input,
					 double &mu_star, double &sigma_star, 
					 GP_Vector &v0, GP_Vector &m0) const
    {
      if(Super::_nu_tilde.Size() == 0 || Super::_tau_tilde.Size() == 0)
	throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

      if(Super::_L.Rows() == 0 || Super::_L.Cols() == 0)
	throw GP_EXCEPTION("Cholesky matrix not computed. "
			   "Run 'Expectation Propagation' first.");
    
      // Compute k_star
      std::list<uint>::const_iterator it = _I.begin(); 
      for(uint i=0; i<_old_d; ++i, ++it);
      GP_Vector k_star(_I.size() - _old_d);
      
      for(uint i=_old_d; it != _I.end(); ++it, ++i)
	k_star[i-_old_d] = Super::CovFunc(Super::GetX()[*it], test_input, Super::_hparms);

      GP_Vector mu_tilde = Super::GetSiteMean().SubVector(_old_d, _d);

      // compute (B^-1 + K)^-1 * k_star using our Cholesky decomp 
      if(_old_d == 0){
	v0 = Super::_L.ForwSubst(k_star);
	m0 = Super::_L.ForwSubst(mu_tilde);

	double kk = Super::CovFunc(test_input, Super::_hparms);
	mu_star    = m0.Dot(v0);
	sigma_star = (kk - v0.Dot(v0));
      }

      else {

	GP_Vector tmp1 = _L3.ForwSubst(mu_tilde - _L2 * m0);
	GP_Vector tmp2 = _L3.ForwSubst(k_star   - _L2 * v0);

	m0.Append(tmp1);	
	v0.Append(tmp2);

	mu_star    += tmp1.Dot(tmp2);
	sigma_star -= tmp2.Dot(tmp2);
      }

      // the class probability we want is the expected value of the 
      // sigmoid under the predictive distribution
      double pi_star = 
	Super::SigFunc(mu_star / sqrt(1./SQR(Super::_lambda) + sigma_star));

      return pi_star;
    }
    

    virtual double PredictionMAP(InputType const &test_input) const 
    {
      if(Super::_nu_tilde.Size() == 0 || Super::_tau_tilde.Size() == 0)
	throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

      if(Super::_L.Rows() == 0 || Super::_L.Cols() == 0)
	throw GP_EXCEPTION("Cholesky matrix not computed. "
			   "Run 'Expectation Propagation' first.");      

      // Compute k_star
      GP_Vector k_star(_I.size());
      uint i=0;
      for(std::list<uint>::const_iterator n = _I.begin(); n != _I.end(); ++n, ++i)
	k_star[i] = Super::CovFunc(Super::GetX()[*n], test_input, Super::_hparms);

      // compute (B^-1 + K)^-1 * k_star using our Cholesky decomp 
      GP_Vector v = Super::_L.SolveChol(k_star);

      // compute predictive mean
      GP_Vector mu_tilde = Super::GetSiteMean();
      uint n = _I.size();
      double mu_star = (mu_tilde + Super::_bias * GP_Vector(n, 1.0)).Dot(v);
    
      double pi_star = Super::SigFunc(mu_star);

      return pi_star;
    }

    static double ComputeBALD(double zval, double mu_star, double sigma_star) 
    {
      // first we compute the normalized entropy
      double nent = -(zval * ::log(zval) + (1.-zval) * ::log(1.-zval)) / LOG2;
      
      double denom = sigma_star + SQR(BALDC); 
      double enumer = BALDC * ::exp(- SQR(mu_star) / (2. * denom));

      return nent - enumer / sqrt(denom);
    }

    double ComputeExpectedInformationGain(InputType const &x_star)
    {
      double mu_star, sigma_star;
      double pred = Prediction(x_star, mu_star, sigma_star);

      double last_mu, last_zeta;
      GetMuZeta(x_star, last_mu, last_zeta);

      double g_kn_pos, nu_kn_pos, g_kn_neg, nu_kn_neg;
      Super::ComputeDerivatives(1, last_mu / last_zeta, 1./last_zeta,
				g_kn_pos, nu_kn_pos);
      double DeltaH_pos = -::log(1.0 - nu_kn_pos * last_zeta) / (2. * LOG2);

      Super::ComputeDerivatives(-1, last_mu / last_zeta, 1./last_zeta,
				g_kn_neg, nu_kn_neg);
      double DeltaH_neg = -::log(1.0 - nu_kn_neg * last_zeta) / (2. * LOG2);
      
      return pred * DeltaH_pos + (1. - pred) * DeltaH_neg;
    }


    /*!
     * Reads everything from an ASCII file
     */
    int Read(std::string filename, int pos = 0)
    {
      int new_pos = Super::Read(filename, pos);
      READ_FILE(ifile, filename.c_str());
      ifile.seekg(new_pos);
      ifile >> _d;

      uint num, idx;
      double val;
      _I.clear();
      for(uint i=0; i<_d; ++i){
	ifile >> idx;
	_I.push_back(idx);
      }
      ifile >> _runEP;
      
      new_pos = _mu.Read(filename, ifile.tellg());
      new_pos = _zeta.Read(filename, new_pos);
      new_pos = _g.Read(filename, new_pos);
      new_pos = _nu.Read(filename, new_pos);
      new_pos = _M.Read(filename, new_pos);

      ifile.seekg(new_pos);
      ifile >> num;
      _point_entropies.clear();
      for(uint i=0; i<num; ++i){
	ifile >> idx >> val;
	_point_entropies.push_back(std::make_pair(idx, val));
      }
      return ifile.tellg();
    }

    /*!
     * Writes everything to an ASCII file
     */
    void Write(std::string filename) const
    {
      Super::Write(filename);
      APPEND_FILE(ofile, filename.c_str());
      ofile << _d << std::endl;
      for(std::list<uint>::const_iterator it = _I.begin(); 
	  it != _I.end(); ++it)
	ofile << *it << " ";
      ofile << std::endl;
      ofile << _runEP << std::endl;
      ofile.close();

      _mu.Write(filename);
      _zeta.Write(filename);
      _g.Write(filename);
      _nu.Write(filename);
      _M.Write(filename);
      APPEND_FILE(ofile1, filename.c_str());
      ofile1 << _point_entropies.size() << std::endl;
      for(std::list<std::pair<uint, double> >::const_iterator it =
	    _point_entropies.begin(); it != _point_entropies.end(); ++it)
	ofile1 << it->first << " " << it->second << "   ";
      ofile1 << std::endl;
    }

  protected:
  
    std::list<uint> _I;
    uint _d, _old_d;
    bool _runEP;
    GP_Vector _mu, _zeta, _g, _nu, _delta, _mu_sqz, _muL0, _muL1, _mu_part;
    GP_Matrix _M, _L2, _L3;
    std::list<std::pair<uint, double> > _point_entropies;

    virtual void ComputeLogZ()
    {
      if(_old_d == 0)
	Super::_logZ = 0;

      for(uint i=_old_d; i<_d; ++i)
      	Super::_logZ -= log(Super::_L[i][i]);

      GP_Vector mu_tilde = Super::GetSiteMean();

      if(_old_d == 0){
	_muL0 = Super::_L.ForwSubst(mu_tilde);
	_muL1 = Super::_L.BackwSubst(_muL0);
	Super::_logZ -= mu_tilde.Dot(_muL1) / 2.;
      }

      else {
	
	// use the Schur complement for efficiency
	mu_tilde = mu_tilde.SubVector(_old_d, _d);

	GP_Vector tmp = _L3.ForwSubst(mu_tilde - _L2 * _muL0);

	_muL0.Append(tmp);

	Super::_logZ -= tmp.Dot(tmp) / 2.;
      }
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
	GP_Vector Cinv_m = Super::_L.SolveChol(Super::GetSiteMean());
	
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

    class LessThan
    {
    public:
      bool operator()(std::pair<uint, double> const &p1,
		      std::pair<uint, double> const &p2) const
      {
	return (p1.second < p2.second);
      }
    };


    template<typename InactiveSetContainer>
    uint FindMostInformativePoint(InactiveSetContainer const &J,
				  GP_Vector const &mu, GP_Vector const &zeta, 
				  double &g_kmax, double &nu_kmax, double &Delta_max, 
				  uint k, uint last = 0)
    {
      typename InactiveSetContainer::iterator argmax = J.begin();
      Delta_max = -HUGE_VAL;

      _point_entropies.clear();

      // loop over the inactive set and see what's interesting there
      for(typename InactiveSetContainer::iterator it_n = J.begin(); 
	  it_n != J.end(); ++it_n){
	
	uint idx = *it_n - last;

	// compute gradient g_kn and nu_kn
	double g_kn, nu_kn;
	Super::ComputeDerivatives(Super::GetY()[*it_n], mu[idx] / zeta[idx],
				  1./zeta[idx], g_kn, nu_kn);
	
	// compute differential entropy score
	// the following line is the original code:
	double DeltaH_kn = -::log(1.0 - nu_kn * zeta[idx]) / (2. * LOG2);

	// however, as we are only interested in the maximum, we can also use this
	// here (and save some log computations):
	//double DeltaH_kn = nu_kn * zeta[idx];

	_point_entropies.push_back(std::make_pair(*it_n, DeltaH_kn));
	
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

      _point_entropies.sort(LessThan());

      return *argmax;
    }

    template<typename InactiveSetContainer>
    uint FindLeastInformativePoint(InactiveSetContainer const &J,
				   GP_Vector const &mu, 
				   GP_Vector const &zeta, 
				   double &g_kmin, double &nu_kmin) const
    {
      typename InactiveSetContainer::iterator argmin = J.begin();
      double Delta_min = -HUGE_VAL;

      // loop over the inactive set and see what's interesting there
      for(typename InactiveSetContainer::iterator it_n = J.begin(); 
	  it_n != J.end(); ++it_n){
	
	// compute gradient g_kn and nu_kn
	double g_kn, nu_kn;
	Super::ComputeDerivatives(Super::GetY()[*it_n], mu[*it_n] / zeta[*it_n],
				  1./zeta[*it_n], g_kn, nu_kn);
	
	// compute differential entropy score
	double DeltaH_kn = -::log(1.0 - nu_kn * zeta[*it_n]) / (2. * LOG2);

	// update minimum and argmin
	if(DeltaH_kn > Delta_min){
	  Delta_min = DeltaH_kn;
	  g_kmin  = g_kn;
	  nu_kmin = nu_kn;
	  argmin  = it_n;
	}
      }

      return *argmin;
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

      //std::cout << "update MLK " << std::endl;
      // update M, L, and K
      UpdateMLK(k, idx, nu, s_nk, M);
    }

    void ComputeSiteParams(GP_Vector const &m, GP_Vector const &beta, 
			   GP_Vector const &mu, std::list<uint>::const_iterator start,
			   uint last_n = 0)
    {
      Super::_nu_tilde.Resize(_I.size());
      Super::_tau_tilde.Resize(_I.size());
      _mu_sqz.Resize(_I.size()); // squeezed version of mu

      uint i=_old_d;
      for(std::list<uint>::const_iterator it = start; 
	  it != _I.end(); ++it, ++i){

	uint idx = *it - last_n;
	Super::_nu_tilde[i]  = m[idx] * beta[idx];
	Super::_tau_tilde[i] = beta[idx];
	_mu_sqz[i] = mu[idx];
      }

      // do EP updates if requested
      if(_runEP){
	std::cout << "doing EP " << std::endl;
	GP_Matrix I = GP_Matrix::Identity(_I.size());
	GP_Matrix Sigma = Super::_K * (I  - Super::_L.SolveChol(Super::_K));
	double delta_max = 1.0;
	uint iter = 0;

	while(delta_max > Super::_EPthresh && iter < Super::_maxEPiter){
	  delta_max = Super::ExpectationPropagation(_mu_sqz, Sigma, _I).Abs().Max();
	  ++iter;
	}
	Super::_L = Super::_K + GP_Matrix::Diag(1./Super::_tau_tilde);
	Super::_L.Cholesky();
      }
    }

  private:

    void UpdateMuZeta(uint k, uint idx, GP_Matrix const &M, 
		      double g_kn, double nu_kn,
		      GP_Vector &s_nk, GP_Vector &mu, GP_Vector &zeta, uint last = 0) const
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

      // update L and K
      if(k == 0){
	if(_runEP){
	  Super::_K = GP_Matrix(1, 1);
	  Super::_K[0][0] = Super::CovFunc(Super::GetX()[idx], Super::_hparms);
	}			 

	Super::_L = GP_Matrix(1, 1);
	Super::_L[0][0] = 1./sqrt_nu_kn;
      }

      else {	  
	if(_runEP){
	  // update K
	  GP_Vector k_vec(k);
	  uint i=0;
	  for(std::list<uint>::const_iterator it = _I.begin(); i < k; ++it, ++i)
	    k_vec[i] = Super::CovFunc(Super::GetX()[*it], Super::GetX()[idx], 
				      Super::_hparms);
	  
	  Super::_K.AppendRow(k_vec); 
	  k_vec.Append(Super::CovFunc(Super::GetX()[idx], Super::_hparms));
	  Super::_K.AppendColumn(k_vec);
	}

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
