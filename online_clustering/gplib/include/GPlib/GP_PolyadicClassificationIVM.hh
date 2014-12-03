#ifndef GP_POLYADIC_CLASSIFICATION_IVM_HH
#define GP_POLYADIC_CLASSIFICATION_IVM_HH

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
   * \class GP_PolyadicClassificationIVM
   * 
   * Implementation of the Informative Vector Machine, a sparse version of 
   * a polyadic GP classifier
   */
  class GP_PolyadicClassificationIVM : 
    public GP_PolyadicClassification<InputType, 
				     GP_CumulativeGaussian, KernelType>
  {
  public:

    typedef GP_DataSet<InputType, uint> GP_DataSetType;
    typedef GP_PolyadicClassificationIVM<InputType, KernelType> Self;
    typedef GP_PolyadicClassification<InputType, GP_CumulativeGaussian, 
				      KernelType> Super;
    typedef typename Super::HyperParameters HyperParameters;

    /*!
     * Default constructor
     */
    GP_PolyadicClassificationIVM() :
      Super(), _d(0)
    {}

    /*!
     * The constructor needs the training data, a number of active points,
     * the slope 'lambda' of the sigmoid, a flag 'runEP' that turns ADF into EP,
     * and the kernel hyper parameters
     */
    GP_PolyadicClassificationIVM(GP_DataSetType const &train_data, uint activeSetSize,
			       double lambda = 1.0, bool runEP = true,
			       HyperParameters const &hparms = HyperParameters()) :
      Super(train_data), _hparms(hparms), _L(), _mu_tilde(), _tau_tilde(),
      _d(activeSetSize), _lambda(lambda), _class2idx(), _idx2class()
    {
      GetClassIndices();
      ComputeBias();
      std::cout << "found " << _idx2class.size() << " classes" << std::endl;
    }

    /*!
     * Checks whether the classname is the correct one
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) || 
	      std::string(classname) == "GP_PolyadicClassificationIVM");
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
      std::cout << "estimation, kernel params: " 
		<< _hparms[0] << " "
		<< _hparms[1] << " "
		<< _hparms[2] << " " << std::endl;

      // Number of points in training data and number of classes
      uint n = Super::Size();
      uint m = _idx2class.size();

      // Initialize site parameters
      std::vector<GP_Vector> mvec(m, GP_Vector(n));
      std::vector<GP_Vector> beta(m, GP_Vector(n));

      // Initialize approximate posterior parameters
      _mu   = std::vector<GP_Vector>(m, GP_Vector(n));
      _zeta = std::vector<GP_Vector>(m, GP_Vector(n));
      for(uint j=0; j<m; ++j)
	for(uint i=0; i<n; ++i)
	  _zeta[j][i] = Super::CovFunc(Super::GetX()[i], Super::GetX()[i], 
				       _hparms);

      // Initialize active and passive set
      _I.resize(m);
      for(uint j=0; j<m; ++j)
	_I[j].clear();
      std::vector<std::set<uint> > J(m);
      for(uint j=0; j<m; ++j)
	for(uint i=0; i<n; ++i)
	  J[j].insert(i);

      _L.clear();
      _L.resize(m);

      _M = std::vector<GP_Matrix>(m, GP_Matrix());
      _g = _nu = std::vector<GP_Vector>(m, GP_Vector(_d));
      
      for(uint k=0; k<_d; ++k){

	// find next point for inclusion into the active set
	uint max_class_idx = 0, max_point_idx = 0;
	double max_delta_val = -HUGE_VAL;
	for(uint j=0; j<m; ++j){

	  double Delta_max;
	  uint max_pnt = FindMostInformativePoint(J[j], j, _mu[j], _zeta[j], 
						  _g[j][k], _nu[j][k], Delta_max);

	  if(Delta_max > max_delta_val){
	    max_delta_val = Delta_max;
	    max_class_idx = j;
	    max_point_idx = max_pnt;
	  }
	}

	// refine site params, posterior params and matrices M and L
	UpdateAll(max_class_idx, max_point_idx, 
		  _g[max_class_idx][k], _nu[max_class_idx][k], 
		  mvec[max_class_idx], beta[max_class_idx], 
		  _mu[max_class_idx], _zeta[max_class_idx], _M[max_class_idx]);

	// add idx to I and remove it from J
	_I[max_class_idx].push_back(max_point_idx);
	J[max_class_idx].erase(max_point_idx);
      }

      // squeeze site means and variances
      _mu_tilde.resize(m);
      _tau_tilde.resize(m);
      for(uint j=0; j<m; ++j){
	_mu_tilde[j] = GP_Vector(_I[j].size());
	_tau_tilde[j] = GP_Vector(_I[j].size());
	uint i=0;
	for(std::list<uint>::const_iterator it = _I[j].begin(); 
	    it != _I[j].end(); ++it, ++i){
	  _mu_tilde[j][i]  = mvec[j][*it];
	  _tau_tilde[j][i] = beta[j][*it];
	}
	//std::cout << "MU: " << _mu_tilde[j] << std::endl;
      }

      // compute log posterior and  derivative wrt the kernel params
      ComputeLogZ();
      ComputeDerivLogZ();
    }

    /*
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
    */

    virtual void UpdateModelParameters(HyperParameters const &new_hyp)
    {
      _hparms = new_hyp;
      
      uint m = _I.size();
      for(uint j=0; j<m; ++j)
	if(_I[j].size() != 0){
	  _L[j] = Super::ComputeCovarianceMatrix(_I[j], _hparms);
	  for(uint i=0; i<_I[j].size(); ++i)
	    _L[j][i][i] += 1./_tau_tilde[j][i];
	  _L[j].Cholesky();
	}

      ComputeLogZ();
      ComputeDerivLogZ();
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
      
      // first we set the initial parameters
      _hparms.FromVector(init);
      for(uint i=0; i<nb_iterations; ++i){

	Estimation();
	_hparms.FromVector(init);
	Optimization(lower_bounds, upper_bounds, residual);

	std::cout << "optimized: " 
		  << _hparms[0] << " " << _hparms[1] << " " 		  
		  << _hparms[2] << " " << std::endl;

	if(residual < min_res){
	  min_res = residual;
	  argmin = _hparms;
	}
      }
      
      _hparms = argmin;
      init = argmin.ToVector();
      Estimation();

      return min_res;
    }

    std::list<uint> const &GetActiveSet(uint class_idx) const
    {
      return _I[class_idx];
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
      for(uint i=0; i<_I.size(); ++i)
      for(std::list<uint>::const_iterator it = _I[i].begin(); 
	  it != _I[i].end(); ++it){
	ofile << *it << " " << Super::GetX()[*it] << " " 
	      << Super::GetY()[*it] << std::endl;
      }
      ofile.close();
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

  /*
    void ExportZeta(char const *filename = "zeta") const
    {
      static uint idx = 0;

      if(_zeta.Size() != Super::Size())
	std::cout << "Zeta is not as long as the training data! "
		  << "Could not export it." << std::endl;

      std::stringstream fname;
      fname << filename << std::setw(3) << std::setfill('0') << idx++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());
      for(uint i=0; i<_zeta.Size(); ++i){
	ofile << _zeta[i] << " "<< Super::GetY()[i] << std::endl;
      }
      ofile.close();
    }
    */
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

    GP_Vector Prediction(InputType const &test_input) const
    {
      GP_Vector mu_star;
      GP_Matrix sigma_star;
      return Prediction(test_input, mu_star, sigma_star);
    }

    virtual GP_Vector Prediction(InputType const &test_input, 
				 GP_Vector &mu_star, GP_Matrix &sigma_star) const
    {
      if(_mu_tilde.size() == 0)
	throw GP_EXCEPTION("Could not do prediction. Run estimation first.");

      if(_L.size() == 0)
	throw GP_EXCEPTION("Cholesky matrices not computed. "
			   "Run 'Expectation Propagation' first.");
    
      double kk = Super::CovFunc(test_input, test_input, _hparms);

      // Number of classes
      uint m = _idx2class.size();
      GP_Vector out(m);
      double sum = 0;

      mu_star =  GP_Vector(m);
      sigma_star =  GP_Matrix(m, m);

      for(uint j=0; j<m; ++j){

	// Compute k_star
	GP_Vector k_star(_I[j].size());
	uint i=0;
	for(std::list<uint>::const_iterator it = _I[j].begin(); 
	    it != _I[j].end(); ++it, ++i)
	  k_star[i] = Super::CovFunc(Super::GetX()[*it], test_input, _hparms);
	
	// compute (B^-1 + K)^-1 * k_star using our Cholesky decomp 
	GP_Vector v = _L[j].SolveChol(k_star);
	
	// compute predictive mean and covariance
	uint n = _I[j].size();
	mu_star[j]    = (_mu_tilde[j] + _bias[j] * GP_Vector(n, 1.0)).Dot(v);
	sigma_star[j][j] = (kk - k_star.Dot(v));
	
	// the class probability we want is the expected value of the 
	// sigmoid under the predictive distribution
	sum += out[j] = Super::SigFunc(mu_star[j] / sqrt(1./SQR(_lambda) + 
							 sigma_star[j][j]));
      }

      for(uint j=0; j<m; ++j)
	out[j] /= sum;

      return out;
    }
    
    static double ComputeBALD(double zval, double mu_star, double sigma_star) 
    {
      // first we compute the normalized entropy
      double nent = -(zval * ::log(zval) + (1.-zval) * ::log(1.-zval)) / LOG2;
      
      double denom = sigma_star + SQR(BALDC); 
      double enumer = BALDC * ::exp(- SQR(mu_star) / (2. * denom));

      return nent - enumer / sqrt(denom);
    }


    /*!
     * Reads everything from an ASCII file
     */
    int Read(std::string filename, int pos = 0)
    {
      /*int new_pos = Super::Read(filename, pos);
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
      return ifile.tellg();*/
    }

    /*!
     * Writes everything to an ASCII file
     */
    void Write(std::string filename) const
    {
      /*Super::Write(filename);
      APPEND_FILE(ofile, filename.c_str());
      ofile << _d << std::endl;
      for(std::list<uint>::const_iterator it = _I.begin(); 
	  it != _I.end(); ++it)
	ofile << *it << " ";
      ofile << std::endl;
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
	ofile1 << std::endl;*/
    }

  private:

    typedef GP_ObjectiveFunction<InputType, Self> ObjectiveFunction;

    std::vector<double> _bias;
    HyperParameters _hparms;
    std::vector<GP_Matrix> _L;
    std::vector<GP_Vector> _mu_tilde, _tau_tilde;
    std::vector<std::list<uint> > _I;
    uint _d;
    double _lambda, _logZ;
    std::vector<GP_Vector> _mu, _zeta, _g, _nu;
    std::vector<GP_Matrix> _M;
    std::list<std::pair<uint, double> > _point_entropies;
    GP_Vector _deriv;

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

    /*!
     * Computes y * nu / (tau * sqrt(1 + 1/tau)) in a numerically stable way
     */
    double ComputeZeroMoment(uint class_label, uint class_idx, double nu, double tau) const
    {
      double c;
      return ComputeZeroMoment(class_label, class_idx, nu, tau, c);
    }

    double ComputeZeroMoment(uint class_label, uint class_idx, double nu, double tau, double &c) const
    {
      std::complex<double> tau_cplx(tau, 0);

      double denom = MAX(abs(sqrt(tau_cplx * 
				  (tau_cplx / SQR(_lambda) + 1.))), 
			 EPSILON);

      int y = (class_label == _idx2class[class_idx] ? 1 : -1);
      c = y * tau / denom;

      if(fabs(nu) < EPSILON)
	return c * _bias[class_idx];

      return y * nu / denom + c * _bias[class_idx];
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


    virtual void ComputeLogZ()
    {
      uint m = _I.size();
      _logZ = 0;

      for(uint j=0; j<m; ++j)
	for(uint i=0; i<_L[j].Cols(); ++i)
	  _logZ -= log(_L[j][i][i]);

      for(uint j=0; j<m; ++j){
	if(_mu_tilde[j].Size() != 0)
	  _logZ -= _mu_tilde[j].Dot(_L[j].SolveChol(_mu_tilde[j])) / 2.;
      }
    }

    /*!
     * Computes the partial derivative of the log-posterior with respect to
     * the  posterior covariance  matrix (K + B^-1).  The function uses the 
     * covariance matrix and the mean value found after point selection,i.e.
     * the true posterior is approximated with the active set.
     */
    GP_Matrix ComputePartDerivCov(uint class_idx) const
    {
      uint n = _I[class_idx].size();
      GP_Matrix I = GP_Matrix::Identity(n);
      GP_Vector Cinv_m = _L[class_idx].SolveChol(_mu_tilde[class_idx]);

      return (GP_Matrix::OutProd(Cinv_m) - _L[class_idx].SolveChol(I)) / 2.; 
    }

    /*!
     * Computes the derivative with respect to the kernel parameters
     */
    void ComputeDerivLogZ()
    {
      _deriv = GP_Vector(_hparms.Size());
      
      // nb of classes
      uint m = _I.size();
      for(uint j=0; j<m; ++j)
	if(_I[j].size() != 0){

	  GP_Matrix Z2 = ComputePartDerivCov(j);
	  
	  for(uint k=0; k<_deriv.Size(); ++k){
	    /*!
	     * CAREFUL HERE! This is a slight hack that only works with kernels where the
	     * last parameter is the data noise. 
	     */
	    if(k == _deriv.Size() - 1)
	      _deriv[k] += Z2.Trace();
	    else {
	      GP_Matrix C = Super::ComputePartialDerivMatrix(_I[j], _hparms, k);
	      _deriv[k] += Z2.ElemMult(C).Sum();	
	    }
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
    uint FindMostInformativePoint(InactiveSetContainer const &J, uint class_idx,
				  GP_Vector const &mu, GP_Vector const &zeta, 
				  double &g_kmax, double &nu_kmax, double &Delta_max)
    {
      typename InactiveSetContainer::iterator argmax = J.begin();
      Delta_max = -HUGE_VAL;

      _point_entropies.clear();

      // loop over the inactive set and see what's interesting there
      for(typename InactiveSetContainer::iterator it_n = J.begin(); 
	  it_n != J.end(); ++it_n){
	
	// compute gradient g_kn and nu_kn
	double g_kn, nu_kn;
	ComputeDerivatives(Super::GetY()[*it_n], class_idx, 
			   mu[*it_n] / zeta[*it_n],
			   1./zeta[*it_n], g_kn, nu_kn);
	
	// compute differential entropy score
	double DeltaH_kn = -::log(1.0 - nu_kn * zeta[*it_n]) / (2. * LOG2);

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

    void UpdateAll(uint class_idx, uint point_idx, 
		   double g, double nu, 
		   GP_Vector &m, GP_Vector &beta,
		   GP_Vector &mu, GP_Vector &zeta,
		   GP_Matrix &M) 
    {
      // update m and beta
      m[point_idx]    = g / nu + mu[point_idx];
      beta[point_idx] = nu / ( 1. - nu * zeta[point_idx]);
      
      // compute zeta and mu
      GP_Vector s_nk, a_nk;
      UpdateMuZeta(point_idx, M, g, nu, s_nk, mu, zeta);

      // update M, L, and K
      UpdateML(class_idx, point_idx, nu, s_nk, M);
    }


    void UpdateMuZeta(uint idx, GP_Matrix const &M, 
		      double g_kn, double nu_kn,
		      GP_Vector &s_nk, GP_Vector &mu, GP_Vector &zeta) const
    {
      uint n = Super::Size();
      GP_Vector k_nk(n);

      for(uint i=0; i<n; ++i)
	k_nk[i] = Super::CovFunc(Super::GetX()[i], Super::GetX()[idx], 
				 _hparms);	  
      if(M.Cols() == 0){
	s_nk = k_nk;
      }
      else {
	s_nk = k_nk - M.Col(idx) * M;
      }
      
      zeta = zeta - nu_kn * s_nk.Sqr();
      mu   = mu + g_kn * s_nk;
    }

    void UpdateML(uint cls_idx, uint pnt_idx, double nu_kn, 
		  GP_Vector const &s_nk, GP_Matrix &M)
    {
      UpdateL(cls_idx, pnt_idx, nu_kn, s_nk, M);
      UpdateM(pnt_idx, nu_kn, s_nk, M);
    }

    void UpdateM(uint idx, double nu_kn, 
		 GP_Vector const &s_nk, GP_Matrix &M) const
    {
      // append sqrt(nu_kmax)s_nk to M_k-1
      double sqrt_nu_kn = sqrt(nu_kn);
      if(M.Cols() == 0){
	M = GP_Matrix(1, s_nk.Size());
	for(uint i=0; i<s_nk.Size(); ++i)
	  M[0][i] = sqrt_nu_kn * s_nk[i];
      }
      else {
	M.AppendRow(sqrt_nu_kn * s_nk);
      }
    }

    void UpdateL(uint cls_idx, uint pnt_idx, double nu_kn, 
		 GP_Vector const &s_nk, GP_Matrix const &M) 
    {
      double sqrt_nu_kn = sqrt(nu_kn);
      GP_Vector a_nk;
      if(pnt_idx < M.Cols())
	a_nk = M.Col(pnt_idx);	

      uint k = _L[cls_idx].Rows();

      // update L
      if(k == 0){
	_L[cls_idx] = GP_Matrix(1, 1);
	_L[cls_idx][0][0] = 1./sqrt_nu_kn;
      }

      else {	  
	GP_Vector extra_col_L = GP_Vector(k+1);
	extra_col_L[k] = 1./sqrt_nu_kn;
	_L[cls_idx].AppendRow(a_nk);
	_L[cls_idx].AppendColumn(extra_col_L);
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
      //GP_Optimizer min(err);
      GP_OptimizerCG min(err, -60);

      uint nb_params = init_params.size();
      std::vector<double> init(nb_params), cur_vals(nb_params);
      
      HyperParameters hparams(init_params);
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
