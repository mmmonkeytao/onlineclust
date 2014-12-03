#ifndef GP_BINARY_CLASSIFICATION_EP_HH
#define GP_BINARY_CLASSIFICATION_EP_HH

#include <iomanip>
#include <complex>

#include "GPlib/GP_Matrix.hh"
#include "GPlib/GP_BinaryClassification.hh"
#include "GPlib/GP_ObjectiveFunction.hh"
#include "GPlib/GP_Optimizer.hh"
#include "GPlib/GP_OptimizerCG.hh"



namespace GPLIB {

  template<typename InputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >
  /*!
   * Standard GP algorithm for binary classification
   *
   * This class implements the Expectation Propagation (EP) algorithm for binary GP
   * classification. The full covariance matrix is calculated and stored, so for 
   * large training sets this class is inefficient.
   *
   * The sigmoid function is fixed to be the cumulative Gaussian. The kernel can be chosen,
   * but it is set to the squared exponential (Gaussian) by default.
   */
  class GP_BinaryClassificationEP : 
    public GP_BinaryClassification<InputType, GP_CumulativeGaussian, KernelType>
				   
  {
  public:

    typedef GP_BinaryClassificationEP<InputType, KernelType> Self;
    typedef GP_BinaryClassification<InputType, 
				    GP_CumulativeGaussian, KernelType> Super;
    typedef typename Super::DataSet          DataSet;
    typedef typename Super::HyperParameters  HyperParameters;

    /*!
     * Default constructor
     */
    GP_BinaryClassificationEP() :
      Super(), _hparms(),
      _K(), _L(), _nu_tilde(), _tau_tilde(), _deriv(), _logZ(0),
       _bias(0), _lambda(1.0), _verbose(true), _EPthresh(1e-4), _maxEPiter(30)
    {}

    /*!
     * The constructor needs a training data set and some 
     * hyper parameters for the kernel. 'lambda' is the slope of the sigmoid.
     */
    GP_BinaryClassificationEP(DataSet const &train_data,
			      std::vector<double> const &hparms = std::vector<double>(),
			      double lambda = 1.0, 
			      GP_Optimizer::OptimizerType type = GP_Optimizer::PR,
			      double step = 0.01, double tol = 1e-2, double eps = 1e-3,
			      bool verbose = true) :
      Super(train_data), _hparms(hparms, train_data.GetInputDim()),
      _K(), _L(), _nu_tilde(), _tau_tilde(), _deriv(), _logZ(0),
       _bias(0), _lambda(lambda),
      _optStep(step), _optTol(tol), _optEps(eps), _optType(type),
      _verbose(verbose), _EPthresh(1e-4), _maxEPiter(60)
    {
      // estimate data bias
      uint class0_size = 0, class1_size = 0;
      for(uint i=0; i<Super::Size(); ++i)
	if(Super::GetY()[i] == 1)
	  ++class0_size;
	else
	  ++class1_size;
     
      if(class0_size == 0 || class0_size == Super::Size())
	throw GP_EXCEPTION("Only one class in training data!");

      //_bias = Super::GetSigFunc().Inv(class0_size / (double)Super::Size());      
    }

    /*!
     * From super class
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      std::string(classname) == "GP_BinaryClassificationEP");
    }

    bool Verbose() const
    {
      return _verbose;
    }

    /*!
     * Assignment operator
     */
    GP_BinaryClassificationEP<InputType, KernelType> const &
    operator=(GP_BinaryClassificationEP<InputType, KernelType> const &other)
    {
      Super::operator=(other);

      _hparms    = other._hparms;
      _K         = other._K;
      _L         = other._L;
      _nu_tilde  = other._nu_tilde;
      _tau_tilde = other._tau_tilde;
      _deriv     = other._deriv;
      _mu        = other._mu;
      _nu_min_z  = other._nu_min_z;
      _s_sqrt    = other._s_sqrt;
      _logZ      = other._logZ;

      _bias      = other._bias;
      _lambda    = other._lambda;
      _optStep   = other._optStep;
      _optTol    = other._optTol;
      _optEps    = other._optEps;
      _optType   = other._optType;
      _verbose   = other._verbose;


      return *this;
    }

    /*!
     * Returns the hyper parameters of the used kernel
     */
    HyperParameters const &GetHyperParams() const
    {
      return _hparms;
    }

    void SetHyperParams(std::vector<double> const &hparms)
    {
      _hparms.FromVector(hparms);
    }

    /*!
     * Runs the Expectation Propagation algorithm until convergence. 
     * The result is a set of site parameters \f$\tilde{nu}\f$, \f$\tilde{tau}\f$,
     * posterior parameters \f$\mu\f$ and \f$\Sigma\f$, as well as
     * a covariance matrix K and a Cholesky decompostion L
     */
    virtual void Estimation()
    {
      // first we compute the covariance matrix
      _K = Super::ComputeCovarianceMatrix(_hparms);

      // initialize the site and posterior params
      uint n = Super::Size();
      _nu_tilde = _tau_tilde = _mu = GP_Vector(n);
      _Sigma = _K;

      // now we run EP until convergence (or max number of iterations reached)
      uint iter = 0;
      double max_delta = 1.;
      do {

	// run EP; the maximum difference in tau is our convergence criterion
	GP_Vector delta_tau = ExpectationPropagation(_mu, _Sigma);
	max_delta = delta_tau.Abs().Max();

	// re-compute mu and Sigma for numerical stability; also computes L
	ComputePosteriorParams();

      } while(max_delta > _EPthresh && ++iter < _maxEPiter);
      
      // Does not happen often, it's not a big problem even if it does. 
      // If it happens too often, just increase '_maxEPiter'
      if(iter == _maxEPiter){
	std::cerr << "Warning! EP did not converge" << std::endl;
      }

      // compute the log posterior and  derivatice wrt. the kernel params
      ComputeLogZ();
      ComputeDerivLogZ();
    }

    /*!
     * Sets the hyperparameters before running Estimation
     */
    void Estimation(HyperParameters const &new_hyp)
    {
      _hparms = new_hyp;
      Estimation();
    }

    /*!
     * Updates the site parameters and the posterior mean and covariance. 
     * In general, this is equal to Estimation(), but derived classes may
     * use this to avoid a full estimation run.
     */
    virtual void UpdateModelParameters(HyperParameters const &new_hyp)
    {
      Estimation(new_hyp);
    }

    /*!
     * Performs one EP update step, i.e. runs once through the data and updates
     * site parameters (nu, tau) and posterior parameters  (mu, Sigma). Returns
     * a vector of delta values, one per data point. These represent the amount
     * of change and can be used as a stopping criterion.
     */
    GP_Vector ExpectationPropagation(GP_Vector &mu, GP_Matrix &Sigma,
				     std::list<uint> const &index_list = std::list<uint>())
    {
      uint n = mu.Size();
      GP_Vector delta_tau(n);
      std::list<uint>::const_iterator it = index_list.begin();

      for(uint i=0; i<n; i++){
	
	// compute approximate cavity parameters
	double tau_min = 1./Sigma[i][i] - _tau_tilde[i];
	double nu_min  = mu[i] / Sigma[i][i] - _nu_tilde[i];

	// compute marginal moments using the derivatives
	double dlZ, d2lZ;
	uint tgt_idx;
	if(index_list.size() == 0)
	  tgt_idx = i; // label index is the next one from the training set
	else if(index_list.size() == n)
	  tgt_idx = *it++; // label index is the next one from the chosen subset
	else
	  throw GP_EXCEPTION("Size of sub-set must match length of posterior "
			     "mean vector. Could not run EP step");

	ComputeDerivatives(Super::GetY()[tgt_idx], nu_min, tau_min, dlZ, d2lZ);

	// update site parameters
	double old_tau_tilde = _tau_tilde[i];
	double denom = 1.0 - d2lZ / tau_min;

	_tau_tilde[i] = MAX(d2lZ / denom, 0);
	_nu_tilde[i]  = (dlZ + nu_min / tau_min * d2lZ) / denom;
	delta_tau[i]  = _tau_tilde[i] - old_tau_tilde;
 
	// update approximate posterior
	GP_Vector si = Sigma.Col(i);

	denom = 1.0 +  delta_tau[i] * si[i];
	//if(fabs(denom) > EPSILON)
	  Sigma -= delta_tau[i] / denom * GP_Matrix::OutProd(si);
	  //else
	  //Sigma -= delta_tau[i] / EPSILON * GP_Matrix::OutProd(si);

	mu = Sigma * _nu_tilde;
      }
      
      // we have a new tau_tilde, lets update the square root of it
      UpdateSqrtS();

      return delta_tau;
    }
    
    /*!
     * Learns hyper parameters from the training data given in the constructor.
     * The optimization  starts with  an inital estimate 'init', and guarantees 
     * that the  parameters are never smaller then 'lower_bound'. The parameter
     * 'nb_iterations' is  only  used in derived  classes. The function returns 
     * the residual that resulted from the optimization.
     */
    virtual double LearnHyperParameters(std::vector<double> &init, 
					GP_Vector lower_bounds = GP_Vector(1), 
					GP_Vector upper_bounds = GP_Vector(1,1.), 
					uint nb_iterations = 0)
    {
      double residual;
      GP_Vector all_lbounds = _hparms.MakeBounds(lower_bounds);
      GP_Vector all_ubounds = _hparms.MakeBounds(upper_bounds);
      Optimization(init, all_lbounds, all_ubounds, residual);
      init = _hparms.ToVector();

      PreparePrediction();

      return residual;
    }

    /*!
     * Returns the covariance matrix
     */
    GP_Matrix const &GetCovMat() const
    {
      return _K;
    }

    /*!
     * Returns the Cholesky decoposition used for the calculation
     */
    GP_Matrix const &GetCholMat() const
    {
      return _L;
    }

    /*!
     * Returns the mean of the latent posterior
     */
    virtual GP_Vector const &GetPosteriorMean() const
    {
      return _mu;
    }

    /*!
     * Returns the mean of the latent posterior
     */
    GP_Matrix const &GetPosteriorCov() const
    {
      return _Sigma;
    }

    /*!
     * Returns the mean values of the likelihood obtained after EP
     */
    GP_Vector GetSiteMean() const
    {
      return _nu_tilde / _tau_tilde;
    }

    /*!
     * Returns the variance values of the likelihood obtained after EP
     */
    GP_Vector GetSiteVar() const
    {
      return 1./_tau_tilde;
    }

    /*!
     * Returns the log-posterior
     */
    double GetLogZ() const
    {
      return _logZ;
    }

    /*!
     * Returns the derivative of the log-posterior with respect to the
     * kernel parameters
     */
    GP_Vector GetDeriv() const
    {
      return _deriv;
    }
    
    /*!
     * Plots the log-posterior in 2D by assigning values between 'min' and 'max' 
     * with a given 'step' to the first two kernel hyper parameters.
     */
    void PlotLogZ(double min, double max, double step)
    {
      HyperParameters store_hparms = _hparms;
      std::vector<double> param_vec = _hparms.ToVector();
      static uint idx_logz = 0;
      std::stringstream fname;

      fname << "logz" << std::setw(3) << std::setfill('0') << idx_logz++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());

      for(param_vec[0] = min; param_vec[0] < max; param_vec[0] += step){
	for(param_vec[1] = min; param_vec[1] < max; param_vec[1] += step){
	  HyperParameters hparms;
	  hparms.FromVector(param_vec);
	  UpdateModelParameters(hparms);

	  ofile << param_vec[0] << " " << param_vec[1] << " " << _logZ << std::endl;
	}
	ofile << std::endl;
      }
      
      ofile.close();
      _hparms = store_hparms;
    }

    void PreparePrediction()
    {
      if(_L.Rows() == 0 || _L.Cols() == 0)
	throw GP_EXCEPTION("Cholesky matrix not computed. "
			   "Run 'Expectation Propagation' first.");

      GP_Vector nu_biased = _nu_tilde + _bias * _tau_tilde;
      _nu_min_z = nu_biased - _s_sqrt * _L.SolveChol(_s_sqrt * (_K * nu_biased));
    }

    double Prediction(InputType const &test_input) const
    {
      double mu_star, sigma_star;
      return Prediction(test_input, mu_star, sigma_star);
    }

    /*!
     * Performs the prediction step for a given 'test_input'. Returns the probability
     * that the test point has label 1, as well as the predictive mean and variance
     */
    virtual double Prediction(InputType const &test_input, 
			      double &mu_star, double &sigma_star) const
    {
      if(_nu_tilde.Size() == 0 || _tau_tilde.Size() == 0)
	throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

      if(_nu_min_z.Size() == 0 || _s_sqrt.Size() == 0){
	std::cerr << _nu_min_z.Size() << std::endl;
	throw GP_EXCEPTION("Could not do prediction. Run PreparePrediction first.");
      }

      uint n=Super::Size();

      // Compute k_star and mu_star
      GP_Vector k_star(n), sk_star(n);
      mu_star = 0;
      for(uint i=0; i<n; i++){
	k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, _hparms);
	sk_star[i] = _s_sqrt[i] * k_star[i]; 
	mu_star += k_star[i] * _nu_min_z[i];
      }
      
      GP_Vector v = _L.ForwSubst(sk_star);
      sigma_star = Super::CovFunc(test_input, test_input, _hparms) - v.Dot(v);
      sigma_star = MAX(sigma_star, 0);

      double pi_star = Super::SigFunc(mu_star / 
				      sqrt(1.0 /SQR(_lambda) + sigma_star));

      return pi_star;
    }


    int Read(std::string filename, int pos = 0)
    {
      int new_pos = Super::Read(filename, pos);

      new_pos = _hparms.Read(filename, new_pos);
      new_pos = _K.Read(filename, new_pos);
      new_pos = _L.Read(filename, new_pos);
      new_pos = _Sigma.Read(filename, new_pos);
      new_pos = _nu_tilde.Read(filename, new_pos);
      new_pos = _tau_tilde.Read(filename, new_pos);
      new_pos = _deriv.Read(filename, new_pos);
      new_pos = _mu.Read(filename, new_pos);
      READ_FILE(ifile, filename.c_str());
      ifile.seekg(new_pos);
      ifile >> _logZ >> _bias >> _lambda;

      return ifile.tellg();
    }

    void Write(std::string filename) const
    {
      Super::Write(filename);
      _hparms.Write(filename);
      _K.Write(filename);
      _L.Write(filename);
      _Sigma.Write(filename);
      _nu_tilde.Write(filename);
      _tau_tilde.Write(filename);
      _deriv.Write(filename);
      _mu.Write(filename);
      APPEND_FILE(ofile, filename.c_str());
      ofile << _logZ << " " << _bias << " " <<  _lambda << std::endl;
    }


  protected:

    HyperParameters _hparms;
    GP_Matrix _K, _L, _Sigma;
    GP_Vector _nu_tilde, _tau_tilde;
    GP_Vector _deriv, _mu, _nu_min_z, _s_sqrt;
    double _logZ, _bias, _lambda;
    double _optStep, _optTol, _optEps;
    GP_Optimizer::OptimizerType _optType;
    bool _verbose;

    const double _EPthresh; // threshold for EP stopping criterion
    const uint _maxEPiter; // maximum number of EP iterations

    /*!
     * Returns true if the optimizer has converged; 'lower_bound' is the 
     * minimal required value of the kernel parameters.
     */
    bool Optimization(GP_Vector const &lower_bounds, 
		      GP_Vector const &upper_bounds, 
		      double &residual)
    {
      return Optimization(_hparms.ToVector(), lower_bounds, 
			  upper_bounds, residual);
    }

    /*!
     * Numerically stable version to compute dlZ and d2lZ
     */
    void ComputeDerivatives(int y, double nu_min, double tau_min,
			    double &dlZ, double &d2lZ) const
    {
      double c;
      double u = ComputeZeroMoment(y, nu_min, tau_min, c);

      dlZ = c * exp(Super::GetSigFunc().LogDeriv(u) - 
		    Super::GetSigFunc().Log(u));

      if(std::isnan(dlZ)){

	std::stringstream msg;
	msg << c << " " << u << " " << tau_min;
	throw GP_EXCEPTION2("dlz is nan: %s", msg.str());
      }

      d2lZ = dlZ * (dlZ + u * c);
    }

    virtual void ComputeLogZ()
    {
      double term1 = 0, term2 = 0, term3 = 0, term4 = 0, term5 = 0;

      GP_Vector sigma_diag = _Sigma.Diag();
      GP_Vector tau_n = 1. / sigma_diag - _tau_tilde;
      GP_Vector nu_n  = _mu / sigma_diag - _nu_tilde;      

      for(uint i=0; i<tau_n.Size(); ++i){
	double zi = ComputeZeroMoment(Super::GetY()[i], nu_n[i], tau_n[i]);
	if(fabs(_tau_tilde[i]) > EPSILON){
	  double arg;
	  if(fabs(tau_n[i]) > EPSILON)
	    arg = 1. + _tau_tilde[i] / tau_n[i];
	  else
	    arg = 1. + _tau_tilde[i] / EPSILON;
	  if(fabs(arg) < EPSILON)
	    arg = EPSILON;
	  term1 += abs(log(std::complex<double>(arg,0)));
	}
	term2 += log(_L[i][i]);
	term3 += Super::GetSigFunc().Log(zi);
	if(fabs(nu_n[i]) > EPSILON){
	    term5 += nu_n[i] * 
	      (_tau_tilde[i] / tau_n[i] * nu_n[i] - 2. * _nu_tilde[i]) /
	      (_tau_tilde[i] + tau_n[i]);
	  }
      }
      
      GP_Matrix SigmaNoDiag = _Sigma;
      for(uint i=0; i<SigmaNoDiag.Cols(); ++i)
	SigmaNoDiag[i][i] = 0.;

      term4 = _nu_tilde.Dot(SigmaNoDiag * _nu_tilde);
      _logZ = term3 - term2 + (term1 + term4 + term5) / 2.;
    }

    virtual void ComputeDerivLogZ()
    {
      GP_Vector b = 
	_nu_tilde - _s_sqrt * _L.SolveChol(_s_sqrt * (_K * _nu_tilde));

      GP_Matrix Z = GP_Matrix::Diag(_s_sqrt), C;
      Z = GP_Matrix::OutProd(b) - Z * _L.SolveChol(Z);

      _deriv = GP_Vector(_hparms.Size());
      for(uint j=0; j<_deriv.Size(); ++j){
	C = Super::ComputePartialDerivMatrix(_hparms, j);
	_deriv[j] = Z.ElemMult(C).Sum() / 2.;
      }
    }

    typedef GP_ObjectiveFunction<InputType, Self> ObjectiveFunction;

    void UpdateSqrtS()
    {
      uint n = _tau_tilde.Size();
      _s_sqrt = GP_Vector(n);

      for(uint i=0; i<n; i++)
	if(_tau_tilde[i] > EPSILON)
	  _s_sqrt[i] = sqrt(_tau_tilde[i]);
	else
	  _s_sqrt[i] = sqrt(EPSILON);
    }

    /*!
     * Computes y * nu / (tau * sqrt(1 + 1/tau)) in a numerically stable way
     */
    double ComputeZeroMoment(int y, double nu, double tau) const
    {
      double c;
      return ComputeZeroMoment(y, nu, tau, c);
    }

    double ComputeZeroMoment(int y, double nu, double tau, double &c) const
    {
      std::complex<double> tau_cplx(tau, 0);

      double denom = MAX(abs(sqrt(tau_cplx * 
				  (tau_cplx / SQR(_lambda) + 1.))), 
			 EPSILON);

      c = y * tau / denom;

      if(fabs(nu) < EPSILON)
	return c * _bias;

      return y * nu / denom + c * _bias;
    }

    /*!
     * Computes mu and Sigma from the site parameters nu_tilde and tau_tilde
     */
    void ComputePosteriorParams()
    {
      uint n = _mu.Size();

      _L = _K;
      _L.CholeskyDecompB(_s_sqrt);

      // Compute S^0.5 * K in place
      _Sigma = GP_Matrix(n, n);
      for(uint i=0; i<n; i++)
	for(uint j=0; j<n; j++)
	  _Sigma[i][j] = _s_sqrt[i] * _K[i][j];
      
      GP_Matrix V = _L.ForwSubst(_Sigma);

      _Sigma = _K - V.TranspTimes(V);
      _mu    = _Sigma * _nu_tilde;
    }

    /*!
     * Returns true if the optimizer has converged
     */
    bool Optimization(std::vector<double> const &init_params, 
		      GP_Vector const &lower_bounds,
		      GP_Vector const &upper_bounds,
		      double &residual)
    {
      if(_verbose){
	std::cout << "init " << std::flush;

	for(uint i=0; i<init_params.size(); ++i)
	  std::cout << init_params[i] << " " << std::flush; 
	std::cout << std::endl;
      }

      ObjectiveFunction err(*this, _hparms.Size(), lower_bounds, upper_bounds);
      //GP_Optimizer min(err, _optType, _optStep, _optTol, _optEps);
      GP_OptimizerCG min(err, -60);

      uint nb_params = init_params.size();
      uint data_dim = Super::GetTrainData().GetInputDim();
      std::vector<double> init(nb_params), cur_vals(nb_params);

      HyperParameters hparams(_hparms);
      hparams.FromVector(init_params);
      hparams.TransformInv(lower_bounds, upper_bounds);

      min.Init(hparams.ToVector());

      if(_verbose)
	std::cout << "starting optimizer..." << std::endl;

      uint nb_max_iter = 50, iter = 0;
      _deriv = GP_Vector(_deriv.Size(), 1.);

      while(min.Iterate() && _deriv.Abs().Max() > 1e-5 && iter < nb_max_iter){

	min.GetCurrValues(cur_vals);
	
	HyperParameters cur_parms(cur_vals, data_dim);
	cur_parms.Transform(lower_bounds, upper_bounds);

	if(_verbose){
	  std::cout << "x " << std::flush;
	  for(uint i=0; i<cur_parms.Size(); ++i)
	    std::cout << cur_parms.ToVector()[i] << " " << std::flush;
	  std::cout << std::endl;
	  std::cout << "deriv " << _deriv << std::endl;
	  std::cout << "val " << min.GetCurrentError() << std::endl;
	}
	++iter;
      }

      bool retval = min.TestConvergence();
      if(_verbose){
	std::cout << std::endl;
	if(retval)
	  std::cout << "converged!" << std::endl;
      }

      if(_verbose){
	min.GetCurrValues(init);
	_hparms.FromVector(init);
	_hparms.Transform(lower_bounds, upper_bounds);

	std::cout << "found parameters: " << std::flush;	
	for(uint i=0; i<_hparms.Size(); ++i){
	  std::cout << _hparms[i] << " " << std::flush;
	}
	std::cout << std::endl;
      }

      residual = min.GetCurrentError();      

      return retval;
    }

  };

}
#endif
