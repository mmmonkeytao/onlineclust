#ifndef GP_POLYADIC_CLASSIFICATION_EP_HH
#define GP_POLYADIC_CLASSIFICATION_EP_HH

#include <list>
#include <set>
#include <limits>
#include <complex>

#include "GPlib/GP_PolyadicClassification.hh"
#include "GPlib/GP_ObjectiveFunction.hh"
#include "GPlib/GP_OptimizerCG.hh"

namespace GPLIB {

  template<typename InputType, 
	   typename KernelType = GP_SquaredExponential<InputType> >

  /*!
   * \class GP_PolyadicClassificationEP
   * 
   * Implementation of the Informative Vector Machine, a sparse version of 
   * a polyadic GP classifier
   */
  class GP_PolyadicClassificationEP : 
    public GP_PolyadicClassification<InputType, 
				     GP_CumulativeGaussian, KernelType>
  {
  public:

    typedef GP_DataSet<InputType, uint> GP_DataSetType;
    typedef GP_PolyadicClassificationEP<InputType, KernelType> Self;
    typedef GP_PolyadicClassification<InputType, GP_CumulativeGaussian, 
				      KernelType> Super;
    typedef typename Super::HyperParameters HyperParameters;

    /*!
     * Default constructor
     */
    GP_PolyadicClassificationEP() :
      Super(), _EPthresh(1e-5), _maxEPiter(50)
    {}

    /*!
     * The constructor needs the training data, a number of active points,
     * the slope 'lambda' of the sigmoid, a flag 'runEP' that turns ADF into EP,
     * and the kernel hyper parameters
     */
    GP_PolyadicClassificationEP(GP_DataSetType const &train_data, 
				std::vector<double> const &hparms = std::vector<double>(), 
				double lambda = 1.0,
				GP_Optimizer::OptimizerType type = GP_Optimizer::PR, 
				double step = 0.01, double tol = 1e-2, double eps = 1e-3) :
      Super(train_data), _hparms(hparms, train_data.GetInputDim()),  
      _alpha(), _beta(),
      _lambda(lambda), _class2idx(), _idx2class(), 
      _EPthresh(1e-5), _maxEPiter(50),
      _optStep(step), _optTol(tol), _optEps(eps), _optType(type)
    {
      GetClassIndices();
      //ComputeBias();
      _bias = std::vector<double>(_idx2class.size(), 0);
      std::cout << "found " << _idx2class.size() << " classes" << std::endl;
    }

    /*!
     * Checks whether the classname is the correct one
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) || 
	      std::string(classname) == "GP_PolyadicClassificationEP");
    }

    /*!
     * Returns the number of classes found in the data
     */
    uint GetNbClasses() const
    {
      return _idx2class.size();
    }

    /*!
     * Computes the active set I, and all other required values 
     * (see base class version of Estimation()).
     */
    virtual void Estimation()
    {
      uint c = _idx2class.size(); // number of classes

      // compute class covariance prior (here equal for all classes)
      _K = Super::ComputeCovarianceMatrix(_hparms);

      // initialize the site and posterior params
      uint n = Super::Size();
      _alpha = std::vector<GP_Vector>(n, GP_Vector(c));
      _beta  = std::vector<GP_Vector>(n, GP_Vector(c));
      _logM0 = std::vector<double>(n, 0);

      _mu    = std::vector<GP_Vector>(n, GP_Vector(c));
      _Sigma = std::vector<GP_Matrix>(n, GP_Matrix(c,c));

      for(uint i=0; i<n; ++i)
	for(uint j=0; j<c; ++j)
	  _Sigma[i][j][j] = _K[i][i];

      _nu_tilde  = std::vector<GP_Vector>(n, GP_Vector(c));
      _Tau_tilde = std::vector<GP_Matrix>(n, GP_Matrix(c,c));
 
      bool last_iter = false, converged = false;
      uint iter = 0;
      double max_delta = 1.;

      // here begins the outer loop
      do {

	std::vector<GP_Matrix> Sigma_i_hat = 
	  ExpectationPropagationOuter(iter, last_iter, _mu, _Sigma);
	
	double sum = 0;
	GP_Matrix Max(c,c);
	for(uint i=0; i<Sigma_i_hat.size(); ++i){

	  GP_Matrix Diff = (_Sigma[i] - Sigma_i_hat[i]).Abs();
	  for(uint j=0; j<c; ++j)	  
	    for(uint k=0; k<c; ++k)
	      Max[j][k] = MAX(Diff[j][k], Max[j][k]);
	}
			       
	max_delta = Max.Fnorm();

	if(last_iter)
	  last_iter = false;
	else {
	  //converged = (max_delta < _EPthresh);
	  converged = (max_delta < 1e-4);
	  if(converged){
	    last_iter = true;
	  }
	}

      } while((!converged && ++iter < _maxEPiter) || last_iter);

      if(converged)
	std::cout << "outer EP converged! " << iter << std::endl;

      std::vector<GP_Matrix> cholAP = ComputePosteriorParams(_mu, _Sigma);
      ComputeLogZ(cholAP);
      ComputeDerivLogZ();
    }


    /*!
     * Performs one EP update step, i.e. runs once through the data and updates
     * site parameters (nu, tau) and posterior parameters  (mu, Sigma). Returns
     * a vector of delta values, one per data point. These represent the amount
     * of change and can be used as a stopping criterion.
     */
    std::vector<GP_Matrix> 
    ExpectationPropagationOuter(uint outer_iter, bool last_iter,
				std::vector<GP_Vector> &mu, std::vector<GP_Matrix> &Sigma)
    {
      uint c = _idx2class.size(); // number of classes
      uint n = Super::Size();
      std::vector<GP_Matrix> Sigma_i_hat(n);
      double df;

      if(outer_iter <= 9) df = 0.85;
      else if(outer_iter <= 14) df = 0.8;
      else if(outer_iter <= 19) df = 0.7;
      else if(outer_iter <= 24) df = 0.6;
      else df = 0.5;

      for(uint i=0; i<n; i++){
	
	uint y_i = Super::GetY()[i];
	
	// compute cavity parameters
	GP_Matrix Sigma_inv = Sigma[i].InverseByChol();
	GP_Matrix Sigma_i = (Sigma_inv - _Tau_tilde[i]).InverseByChol();
	GP_Vector mu_i = Sigma_i * (Sigma_inv * mu[i] - _nu_tilde[i]);

	GP_Vector alpha_before = _alpha[i];
	GP_Vector beta_before  = _beta[i];

	// here begins the inner loop
	std::pair<GP_Vector, GP_Vector> alpha_beta = 
	  ExpectationPropagationInner(y_i, last_iter, mu_i, Sigma_i, _logM0[i]);

	// update site params
	_alpha[i] = alpha_before + df * (alpha_beta.first - alpha_before);
	_beta[i]  = beta_before  + df * (alpha_beta.second - beta_before);

	double one_over_sum_alpha = 1./_alpha[i].Sum();
	Sigma_i_hat[i] = Sigma_i;
	_Tau_tilde[i]  = (GP_Matrix::Diag(_alpha[i]) - 
			  one_over_sum_alpha * GP_Matrix::OutProd(_alpha[i]));
	_nu_tilde[i] = _beta[i].Sum() * one_over_sum_alpha * _alpha[i] - _beta[i]; 
      }      

      ComputePosteriorParams(mu, Sigma);

      return Sigma_i_hat;
    }

    std::pair<GP_Vector, GP_Vector>
    ExpectationPropagationInner(uint y_i, bool last_iter,
				GP_Vector &mu_i, GP_Matrix &Sigma_i, double &lm0) const
    {
      uint c = mu_i.Size();
      GP_Vector delta_alpha(c), delta_beta(c);
      GP_Vector b(c+1), z_hat(c,1);
      b[y_i] = b[c] = 1;
      
      // Append auxiliary variable
      mu_i.Append(0);
      Sigma_i.AppendColumn(GP_Vector(c));
      Sigma_i.AppendRow(GP_Vector(c+1));
      Sigma_i[c][c] = 1.;

      // store prior mu and Sigma
      GP_Matrix Sigma0 = Sigma_i;
      GP_Vector mu0 = mu_i;
      
      // initialize alpha and beta
      GP_Vector alpha(c), beta(c);
      alpha[y_i] = 1;
      GP_Vector vcvec(c, 1);
      GP_Vector mcvec(c, 0);
      
      uint inner_iter = 0;
      double max_delta_alpha = 1, max_delta_beta = 1;
      do {
	
	for(uint j=0; j<c; ++j)
	  if(j != y_i){

	    GP_Vector bcopy = b;
	    bcopy[j] = -1;
	    
	    double v = bcopy.Dot(Sigma_i * bcopy);
	    double m = bcopy.Dot(mu_i);
	      
	    // compute approximate cavity parameters
	    double tau_min = 1./v - alpha[j];
	    double nu_min  = m / v - beta[j];

	    vcvec[j] = 1./tau_min;
	    mcvec[j] = vcvec[j] * nu_min;

	    // compute marginal moments
	    double m_hat, v_hat;
	    ComputeMoments(y_i, nu_min, tau_min, m_hat, v_hat, z_hat[j]);
	      
	    // update site parametes
	    delta_alpha[j] = 1./v_hat - 1./v;
	    delta_beta[j]  = m_hat / v_hat - m / v;
	    alpha[j] += delta_alpha[j];
	    beta[j]  += delta_beta[j];
	      
	    // rank-1 update
	    GP_Vector theta = Sigma_i.Col(y_i) - Sigma_i.Col(j) + Sigma_i.Col(c);
	    double denom = 1. + delta_alpha[j] * v;
	    Sigma_i -= delta_alpha[j] / denom * GP_Matrix::OutProd(theta);
	    mu_i    += (delta_beta[j] - delta_alpha[j] * m) / denom * theta;
	  }

	max_delta_alpha = delta_alpha.Abs().Sum();
	max_delta_beta  = delta_beta.Abs().Sum();
      
      } while((max_delta_alpha > _EPthresh || max_delta_beta > _EPthresh ) && 
	      ++inner_iter < _maxEPiter);
      
      //if(inner_iter < _maxEPiter)
      //std::cout << "   inner EP converged! " << inner_iter << std::endl;

      // compute the log marginal in the last iteration
      if(last_iter)
	lm0 = ComputeLogMarginal(y_i, z_hat, vcvec, mcvec, 
				 mu0, Sigma0, mu_i, Sigma_i);

      // Remove auxiliary variable
      mu_i.RemoveLast();
      Sigma_i = Sigma_i.RemoveRowAndColumn(c);

      return std::make_pair(alpha, beta);
    }

    virtual void UpdateModelParameters(HyperParameters const &new_hyp)
    {
      _hparms = new_hyp;
      
      Estimation();
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
      double residual;
      Optimization(init, lower_bounds, upper_bounds, residual);
      init = _hparms.ToVector();
      return residual;
    }

    double GetLogZ() const
    {
      return _logZ;
    }

    GP_Vector GetDeriv() const
    {
      return _deriv;
    }

    uint GetLabel(uint class_idx) const
    {
      if(_idx2class.size() == 0)
	throw GP_EXCEPTION("No class indices given");

      return _idx2class[class_idx];
    }


    /*!
     * Plots the log-posterior in 2D by assigning values between 'min' and 'max' 
     * with a given 'step' to the first two kernel hyper parameters.
     */
    void PlotClassif(double min, double max, double step)
    {
      InputType input(2);
      static uint idx_clsf = 0;
      std::stringstream fname;

      fname << "clsf" << std::setw(3) 
	    << std::setfill('0') << idx_clsf++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());

      for(input[0] = min; input[0] < max; input[0] += step){
	for(input[1] = min; input[1] < max; input[1] += step){

	  ofile << input[0] << " " << input[1] << " " 
		<< Prediction(input) << std::endl;
	}
	ofile << std::endl;
      }
      
      ofile.close();
    }

    GP_Vector Prediction(InputType const &test_input) const
    {
      GP_Vector mu_star;
      GP_Matrix sigma_star;
      return Prediction(test_input, mu_star, sigma_star);
    }

    virtual GP_Vector Prediction(InputType const &test_input, 
				 GP_Vector &mu_star, GP_Matrix &Sigma_star) const
    {
      if(_alpha.size() == 0 or _beta.size() == 0)
	throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

      double kk = Super::CovFunc(test_input, test_input, _hparms);      
      uint n = Super::Size();
      // Number of classes
      uint c = _idx2class.size();
      GP_Vector out(c);

      // Compute k_star
      GP_Vector k_star(n);
      for(uint i=0; i<n; ++i)
	k_star[i] = Super::CovFunc(Super::GetX()[i], test_input, _hparms);

      // compute predictive mean 
      mu_star = k_star * _Spostnu;

      // compute predictive covariance
      std::vector<GP_Vector> invcholPBKt(c);
      for(uint j=0; j<c; ++j)
	invcholPBKt[j] = _cholP.ForwSubst(_B[j] * k_star);
      
      Sigma_star = GP_Matrix(c,c);
      for(uint j=0; j<c; ++j){
	double kBk = k_star.Dot(_B[j] * k_star);
	
	Sigma_star[j][j] = kk - kBk + invcholPBKt[j].Dot(invcholPBKt[j]);
	for(uint k=j+1; k<c; ++k)
	  Sigma_star[j][k] = Sigma_star[k][j] = invcholPBKt[j].Dot(invcholPBKt[k]);
      }

      GP_Vector mu_hat;
      GP_Matrix Sigma_hat;
      GP_Vector lm0(c);
      double sum = 0;
      for(uint j=0; j<c; ++j){
	mu_hat    = mu_star;
	Sigma_hat = Sigma_star;
	std::pair<GP_Vector, GP_Vector> alpha_beta = 
	  ExpectationPropagationInner(j, true, mu_hat, Sigma_hat, lm0[j]);

	sum += out[j] = exp(lm0[j]);
      }

      for(uint j=0; j<c; ++j)
	out[j] /= sum;

      return out;
    }
    
  private:

    typedef GP_ObjectiveFunction<InputType, Self> ObjectiveFunction;

    std::vector<double> _bias;
    HyperParameters _hparms;
    GP_Matrix _K, _cholP, _Spostnu;
    std::vector<GP_Matrix> _Tau_tilde, _Sigma, _B;
    std::vector<GP_Vector> _alpha, _beta, _nu_tilde, _mu,  _BKnu;
    GP_Vector _invPBKnu;
    std::vector<double> _logM0;
    double _lambda, _logZ;
    GP_Vector _deriv;
    double _optStep, _optTol, _optEps;
    GP_Optimizer::OptimizerType _optType;

    double _EPthresh; // threshold for EP stopping criterion
    uint _maxEPiter; // maximum number of EP iterations

    std::map<uint, uint> _class2idx;
    std::vector<uint> _idx2class;

   void GetClassIndices()
    {
      _class2idx.clear();
      _idx2class.clear();

      std::map<uint, uint>::iterator it;

      for(uint i=0; i<Super::Size(); ++i){
	
	uint y = Super::GetY()[i];
	it = _class2idx.find(y);
	if(it == _class2idx.end()){
	  _class2idx[y] = _class2idx.size();
	  _idx2class.push_back(y);
	}
      }
    }


    void ComputeBias()
    {
      // Number of classes
      uint m = _idx2class.size();

      std::vector<uint> class_freq(m);
      for(uint i=0; i<Super::Size(); ++i)
	for(uint j=0; j<m; ++j)
	  if(Super::GetY()[i] == _idx2class[j]){
	    ++class_freq[j];
	    break;
	  }

      _bias.resize(m);
      for(uint j=0; j<m; ++j)
	_bias[j] = Super::GetSigFunc().Inv(class_freq[j] / (double)Super::Size());

      std::cout << "bias is " << std::endl;
      for(uint j=0; j<m; ++j)
	std::cout << _bias[j] << " ";
      std::cout << std::endl;
    }

    GP_Vector MakeSqrtS(GP_Vector const &alpha) const
    {
      uint c = alpha.Size();
      GP_Vector s_sqrt(c);

      for(uint j=0; j<c; j++)
	if(alpha[j] > EPSILON)
	  s_sqrt[j] = sqrt(alpha[j]);
	else
	  s_sqrt[j] = sqrt(EPSILON);

      return s_sqrt;
    }

    GP_Matrix MakeBtilde(uint y_i) const
    {
      uint c = _idx2class.size();
      GP_Matrix Btilde(c+1, c-1);
      uint offs = 0;

      for(uint j=0; j<c-1; ++j){
	Btilde[y_i][j] = Btilde[c][j] = 1;
	for(uint k=j+offs; k<c; ++k){
	  if(k != y_i){
	    Btilde[k][j] = -1;
	    break;
	  }
	  else offs = 1;
	}
      }
      return Btilde;
    }

   /*!
     * Computes mu and Sigma from the (natural) site parameters alpha and beta
     */
    std::vector<GP_Matrix> 
    ComputePosteriorParams(std::vector<GP_Vector> &mu, std::vector<GP_Matrix> & Sigma)
    {
      uint n = _alpha.size();
      if(n == 0) return std::vector<GP_Matrix>();

      uint c = _alpha[0].Size(); // number of classes
      std::vector<GP_Matrix> cholAP;

      _cholP = GP_Matrix(n, n);
      _B.resize(c);

      for(uint j=0; j<c; ++j){
	GP_Vector pi_j(n);
	for(uint i=0; i<n; ++i){
	  uint y_i = Super::GetY()[i];
	  if(y_i == j)
	    pi_j[i] = 1;
	  else
	    pi_j[i] = _alpha[i][j];
	}

	GP_Vector s_sqrt = MakeSqrtS(pi_j);
	GP_Matrix L = _K;
	L.CholeskyDecompB(s_sqrt);
	cholAP.push_back(L);

	GP_Matrix invcholADsq = L.ForwSubst(GP_Matrix::Diag(s_sqrt));
	_B[j] = invcholADsq.TranspTimes(invcholADsq);
	
	_cholP += _B[j];
      }
      
      _cholP.Cholesky();
      cholAP.push_back(_cholP);

      GP_Vector BKnu_sum(n);
      std::vector<GP_Vector> Knu(c), nu_tilde_t(c, GP_Vector(n));
      std::vector<GP_Matrix> BK(c), invcholPBK(c);

      _BKnu.resize(c);
      for(uint j=0; j<c; ++j){
	for(uint i=0; i<n; ++i)
	  nu_tilde_t[j][i] = _nu_tilde[i][j];
	BK[j] = _B[j] * _K;
	_BKnu[j] = BK[j] * nu_tilde_t[j];
	BKnu_sum += _BKnu[j];
	invcholPBK[j] = _cholP.ForwSubst(BK[j]);
      }
      _invPBKnu = _cholP.SolveChol(BKnu_sum);

      mu = std::vector<GP_Vector>(n, GP_Vector(c));
      for(uint j=0; j<c; ++j){
	GP_Vector mu_j = _K * (nu_tilde_t[j] - _BKnu[j] + _B[j] * _invPBKnu);
	for(uint i=0; i<n; ++i)
	  mu[i][j] = mu_j[i];
      }


      Sigma = std::vector<GP_Matrix>(n, GP_Matrix(c,c));
      for(uint j1=0; j1<c; ++j1){
	GP_Matrix KBK = _K.ElemMult(BK[j1]);
	GP_Matrix PBK = invcholPBK[j1].ElemMult(invcholPBK[j1]);
	GP_Vector sum1(n), sum2(n);
	for(uint i=0; i<n; ++i){
	  sum1 += KBK.Row(i);
	  sum2 += PBK.Row(i);
	}
	GP_Vector Sigma_kk = _K.Diag() - sum1 + sum2; 
	
	for(uint i=0; i<n; ++i)
	  Sigma[i][j1][j1] = Sigma_kk[i];

	for(uint j2 = j1+1; j2 < c; ++j2){
	  GP_Matrix PBK2 = invcholPBK[j1].ElemMult(invcholPBK[j2]);

	  GP_Vector sum3(n);
	  for(uint i=0; i<n; ++i)
	    sum3 += PBK2.Row(i);
	  for(uint i=0; i<n; ++i)
	    Sigma[i][j1][j2] = Sigma[i][j2][j1] = sum3[i];
	}
      }

      return cholAP;
    }

    double ComputeLogMarginal(uint y_i, GP_Vector const &z_hat,
			      GP_Vector const &vcvec, GP_Vector const &mcvec,
			      GP_Vector const &mu0,  GP_Matrix const &Sigma0,
			      GP_Vector const &mu_i, GP_Matrix const &Sigma_i) const 
			      
    {
      GP_Matrix Btilde = MakeBtilde(y_i);
      GP_Vector mivec  = mu_i * Btilde;
      GP_Vector vivec  = Btilde.ElemMult(Sigma_i * Btilde).RowSum();
      
      //std::cout << "mivec " << mivec << std::endl;
      //std::cout << "vivec " << vivec << std::endl;
      
      //std::cout << "mcvec " << mcvec << std::endl;
      //std::cout << "vcvec " << vcvec << std::endl;
      
      GP_Matrix cholSigma = Sigma_i;
      cholSigma.Cholesky();
      //std::cout << Sigma_i << std::endl;
      //std::cout << cholSigma << std::endl;
      
      GP_Vector iLm = cholSigma.ForwSubst(mu_i);
      GP_Matrix cholSigma0 = Sigma0;
      cholSigma0.Cholesky();
      GP_Vector iLmprior = cholSigma0.ForwSubst(mu0);
      
      double t1 = iLm.Dot(iLm) / 2.;
      double t2 = cholSigma.SumLogDiag();
      double t3 = iLmprior.Dot(iLmprior) / 2.;
      double t4 = cholSigma0.SumLogDiag();
      double t5 = z_hat.Log().Sum();
      double t6 = (0.5 * mcvec.Sqr() / vcvec +  0.5 * vcvec.Log()).Sum() + 
	( -0.5 * mivec.Sqr() / vivec - 0.5 * vivec.Log()).Sum();
      
      //std::cout << t1 << " " << t2 << " " << t3 << " " 
      //	<< t4 << " " << t5 << " " << t6 << std::endl;
      return t1 + t2 - t3 - t4 + t5 + t6;      
    }

    void ComputePosteriorParams(uint i, 
				GP_Vector const &alpha, GP_Vector const &beta,
				GP_Vector & mu, GP_Matrix & Sigma)
    {
      uint c = alpha.Size(); // number of classes
      GP_Vector s_sqrt = MakeSqrtS(alpha);

      GP_Matrix K = GP_Matrix::Diag(_K[i][i] * GP_Vector(c, 1));
      GP_Matrix L = K;
      L.CholeskyDecompB(s_sqrt);

      // Compute S^0.5 * K in place
      Sigma = GP_Matrix(c, c);
      for(uint i=0; i<c; i++)
	for(uint j=0; j<c; j++)
	  Sigma[i][j] = s_sqrt[i] * K[i][j];
      
      GP_Matrix V = L.ForwSubst(Sigma);

      Sigma = K - V.TranspTimes(V);
      mu    = Sigma * beta;
    }


    /*!
     * Numerically stable version to compute dlZ and d2lZ
     */
    void ComputeMoments(uint y, double nu_min, 
			double tau_min,
			double &m_hat, double &v_hat, double &z_hat) const
    {
      uint c = _idx2class.size(); // number of classes
      uint class_idx = _class2idx.find(y)->second;

      double z_j, rho_j, gamma_j;
      double d;
      z_j = ComputeZeroMoment(class_idx, nu_min, tau_min, d);

      rho_j = d * exp(Super::GetSigFunc().LogDeriv(z_j) - 
		      Super::GetSigFunc().Log(z_j));
      
      if(std::isnan(rho_j)){
	  
	std::stringstream msg;
	msg << z_j << " " << rho_j << " " << tau_min;
	throw GP_EXCEPTION2("dlz is nan: %s", msg.str());
      }
      
      gamma_j = rho_j * rho_j + rho_j * z_j * d;
      
      m_hat = (rho_j + nu_min ) / tau_min;
      v_hat = (tau_min - gamma_j) / SQR(tau_min);
      z_hat = Super::GetSigFunc()(z_j);
    }

    /*!
     * Computes y * nu / (tau * sqrt(1 + 1/tau)) in a numerically stable way
     */
    double ComputeZeroMoment(uint class_idx, double nu, double tau) const
    {
      double c;
      return ComputeZeroMoment(class_idx, nu, tau, c);
    }

    double ComputeZeroMoment(uint class_idx, double nu, double tau, double &c) const
    {
      std::complex<double> tau_cplx(tau, 0);

      double denom = MAX(abs(sqrt(tau_cplx * 
				  (tau_cplx / SQR(_lambda) + 1.))), 
			 EPSILON);

      c = tau / denom;

      if(fabs(nu) < EPSILON)
	return c * _bias[class_idx];

      return nu / denom + c * _bias[class_idx];
    }


    /*!
     * Numerically stable version to compute dlZ and d2lZ
     * 'bin_label' is 1 for class and -1 for non-class
     */
    void ComputeDerivatives(uint class_label, uint class_idx, 
			    double nu_min, double tau_min,
			    double &dlZ, double &d2lZ) const
    {
      double c;
      double u = ComputeZeroMoment(class_label, class_idx, 
				   nu_min, tau_min, c);

      dlZ = c * exp(Super::GetSigFunc().LogDeriv(u) - 
		    Super::GetSigFunc().Log(u));

      if(std::isnan(dlZ)){
	std::stringstream msg;
	msg << c << " " << u << " " << tau_min;
	throw GP_EXCEPTION2("dlz is nan: %s", msg.str());
      }

      d2lZ = dlZ * (dlZ + u * c);
    }


    virtual void ComputeLogZ(std::vector<GP_Matrix> const &cholAP)
    {
      uint n = Super::Size();
      uint c = _idx2class.size();
      double logZcav = 0, logZmarg = 0;

      for(uint i=0; i<n; ++i){

	GP_Matrix cholSigma = _Sigma[i];
	cholSigma.Cholesky();
	GP_Matrix icholSigma = cholSigma.ForwSubst(GP_Matrix::Identity(c));
	GP_Matrix invSigma = icholSigma.TranspTimes(icholSigma);

	GP_Matrix Tau_i = invSigma - _Tau_tilde[i];
	GP_Matrix cholTau_i = Tau_i;
	cholTau_i.Cholesky();
	GP_Matrix icholTau = cholTau_i.ForwSubst(GP_Matrix::Identity(c));
	GP_Vector nu_i = invSigma * _mu[i] - _nu_tilde[i];

	GP_Matrix Sigma_i = icholTau.TranspTimes(icholTau);
	GP_Vector mu_i = Sigma_i * nu_i;
	GP_Vector cholSigmamu = cholSigma.ForwSubst(_mu[i]);

	//std::cout << mu_i << std::endl;
	//std::cout << Sigma_i << std::endl << std::endl;	
	//std::cout << Tau_i << std::endl << std::endl;
	//std::cout << cholSigmamu << std::endl << std::endl;
	
	logZcav  += 0.5 * mu_i.Dot(Tau_i*mu_i) - cholTau_i.SumLogDiag();
	logZmarg -= 0.5 * cholSigmamu.Dot(cholSigmamu) + cholSigma.SumLogDiag();

	//std::cout << "logzcav  " << logZcav << std::endl;
	//std::cout << "logzmarg " << logZmarg << std::endl;
      }
      
      double logdetAP = 0;
      for(uint j=0; j<cholAP.size(); ++j)
	logdetAP += cholAP[j].SumLogDiag();

      //std::cout << "logdetAP " << logdetAP << std::endl;
      double logdetRDR = 0;
      double mu_times_nu = 0;
      double sumLogM0 = 0;
      for(uint i=0; i<n; ++i){
	double rowsum = 0;
	for(uint j=0; j<c; ++j){
	  rowsum += _alpha[i][j];
	}
	logdetRDR   += log(rowsum);
	mu_times_nu += _mu[i].Dot(_nu_tilde[i]);
	sumLogM0    += _logM0[i];
      }

      _logZ = 0.5 * mu_times_nu - logdetAP + logdetRDR + sumLogM0 + logZcav + logZmarg;
    }

    /*!
     * Computes the partial derivative of the log-posterior with respect to
     * the  posterior covariance  matrix (K + B^-1).  The function uses the 
     * covariance matrix and the mean value found after point selection,i.e.
     * the true posterior is approximated with the active set.
     */
    GP_Matrix ComputePartDerivCov(uint class_idx) const
    {
      /*      uint n = _I[class_idx].size();
      GP_Matrix I = GP_Matrix::Identity(n);
      GP_Vector Cinv_m = _L[class_idx].SolveChol(_mu_tilde[class_idx]);

      return (GP_Matrix::OutProd(Cinv_m) - _L[class_idx].SolveChol(I)) / 2.; */
      return GP_Matrix();
    }

    /*!
     * Computes the derivative with respect to the kernel parameters
     */
    void ComputeDerivLogZ()
    {
      uint n = Super::Size();
      uint c = _idx2class.size();

      _deriv = GP_Vector(_hparms.Size());

      _Spostnu = GP_Matrix(n, c);
      for(uint j=0; j<c; ++j){
	GP_Vector temp = (_BKnu[j] - _B[j] * _invPBKnu);
	for(uint i=0; i<n; ++i){
	  _Spostnu[i][j] = _nu_tilde[i][j] - temp[i];
	}
      }

      std::vector<GP_Matrix> invcholPB(c);
      for(uint j=0; j<c; ++j)
	invcholPB[j] = _cholP.ForwSubst(_B[j]);

      GP_Matrix Z(n,n);
      for(uint j=0; j<c; ++j)
	Z += _B[j] - invcholPB[j].TranspTimes(invcholPB[j]);

      Z = _Spostnu.TimesTransp(_Spostnu) - Z;

      for(uint k=0; k<_deriv.Size(); ++k){
	
	if(k == _deriv.Size() - 1){
	  _deriv[k] = Z.Trace() / 2.;
	  std::cout << "deriv " << _deriv[k] << std::endl;
	}
	else {
	  GP_Matrix C = Super::ComputePartialDerivMatrix(_hparms, k);
	  _deriv[k] = Z.ElemMult(C).Sum() / 2.;
	  std::cout << "deriv " << _deriv[k] << std::endl;
	}
      }
    }


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
     * Returns true if the optimizer has converged
     */
    bool Optimization(std::vector<double> const &init_params, 
		      GP_Vector const &lower_bounds,
		      GP_Vector const &upper_bounds,
		      double &residual)
    {
      std::cout << "init " << init_params[0] << " " 
		<< init_params[1] << " " << init_params[2] << std::endl;

      ObjectiveFunction err(*this, _hparms.Size(), lower_bounds, upper_bounds);
      //GP_Optimizer min(err, _optType, _optStep, _optTol, _optEps);
      GP_OptimizerCG min(err, -60);

      uint nb_params = init_params.size();
      std::vector<double> init(nb_params), cur_vals(nb_params);
      
      HyperParameters hparams(init_params);
      //hparams = hparams.TransformInv(lower_bounds, upper_bounds);
      hparams.TransformInv(lower_bounds, upper_bounds);

      min.Init(hparams.ToVector());
      std::cout << "starting optimizer..." << std::endl;
      uint nb_max_iter = 50, iter = 0;
      _deriv = GP_Vector(_deriv.Size(), 1.);

      char bar = '-';
      while(min.Iterate() && _deriv.Abs().Max() > 1e-5 && iter < nb_max_iter){

	std::cout << bar << "\r" << std::flush;
	if(bar == '-')
	  bar = '\\';
	else if(bar == '\\')
	  bar = '|';
	else if(bar == '|')
	  bar = '/';
	if(bar == '/')
	  bar = '-';

	min.GetCurrValues(cur_vals);
	HyperParameters cur_parms(cur_vals);
	//cur_parms = cur_parms.Transform(lower_bounds, upper_bounds);
	cur_parms.Transform(lower_bounds, upper_bounds);
	std::cout << GP_Vector(cur_parms.ToVector()) << std::endl;
	std::cout << "err: " << min.GetCurrentError() << std::endl;
	++iter;
      }
      std::cout << std::endl;
      bool retval = min.TestConvergence();
      if(retval)
	std::cout << "converged!" << std::endl;

      min.GetCurrValues(init);
      residual = min.GetCurrentError();

      std::cout << "found parameters: " << std::flush;
      _hparms = HyperParameters(init);
      _hparms.Transform(lower_bounds, upper_bounds);

      for(uint i=0; i<_hparms.Size(); ++i){
	std::cout << _hparms[i] << " " << std::flush;
      }
      std::cout << std::endl;

      return retval;
    }

  };
  
}
#endif
