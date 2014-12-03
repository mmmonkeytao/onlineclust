#ifndef GP_BINARY_CLASSIFICATION_MI_HH
#define GP_BINARY_CLASSIFICATION_MI_HH

#include <list>
#include <set>
#include <limits>

#include <boost/pending/relaxed_heap.hpp>
#include <boost/property_map/vector_property_map.hpp>
#include <boost/pending/indirect_cmp.hpp>

#include "GPlib/GP_BinaryClassificationIVM.hh"

namespace GPLIB {
//	class max_extvalue
//	{
//	public:
//		typedef bool result_type;
//
//		max_extvalue(const std::vector<double>& values) : values(values) {};
//
//		bool operator()(size_t x, size_t y) const {
//			assert(values[x] && values[y]);
//			return values[x] > values[y];
//		}
//	protected:
//		const std::vector<double>& values;
//	};

  template<typename InputType, 
	   template <typename> class CovarianceFuncType = GP_SquaredExponential>

  /*!
   * \class GP_BinaryClassificationMIApprox
   * 
   * Implementation of the Informative Vector Machine, a sparse version of 
   * a binary GP classifier
   */
  class GP_BinaryClassificationMIApprox :
    public GP_BinaryClassificationIVM<InputType, CovarianceFuncType>
  {
  public:

    typedef GP_BinaryClassificationIVM<InputType, CovarianceFuncType> Super;
    typedef typename Super::DataSet DataSet;
    typedef typename Super::HyperParameters HyperParameters;

    typedef boost::vector_property_map<double> values_type;
    typedef boost::indirect_cmp<values_type,std::greater<double> > Cmp;

    typedef boost::relaxed_heap<size_t, Cmp> HeapType;

    /*!
     * Default constructor
     */
    GP_BinaryClassificationMIApprox() : _delta_values(0), _cmp(_delta_values, std::greater<double>()), _heap(0, _cmp), _CovThresh(0.1), Super()
    { }

    /*!
     * The constructor needs the training data, a number of active points,
     * the slope 'lambda' of the sigmoid, a flag 'runEP' that turns ADF into EP,
     * and the kernel hyper parameters
     */
    GP_BinaryClassificationMIApprox(DataSet const &train_data, uint activeSetSize,
			       double lambda = 1.0, bool runEP = true,
			       HyperParameters const &hparms = HyperParameters()) :
			      	 _delta_values(train_data.Size()), _cmp(_delta_values, std::greater<double>()), _heap(train_data.Size(), _cmp), _CovThresh(0.95 * hparms[0]), Super(train_data, activeSetSize, lambda, runEP, hparms)
    {
      for(uint i=0; i<Super::Size(); ++i) {
      	_J.push_back(i);
      }
      //_delta_values.resize(_J.size(), std::numeric_limits<double>::min());
    }

    /*!
     * Checks whether the classname is the correct one
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) || 
	      std::string(classname) == "GP_BinaryClassificationMIApprox");
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

      _CovThresh = 0.95 * Super::_hparms[0];
      std::cout << "setting covariance threshold to " << _CovThresh << std::endl;

      // Number of points in training data
      uint n = Super::Size();

      // Initialize site parameters
      GP_Vector m(n);
      GP_Vector beta(n);

      // Initialize approximate posterior parameters
      Super::_mu   = GP_Vector(n);
      Super::_zeta = GP_Vector(n);
      for(uint i=0; i<n; ++i) {
      	Super::_zeta[i] = Super::CovFunc(Super::GetX()[i], Super::GetX()[i], Super::_hparms);
      }

      Super::_M = GP_Matrix();
      Super::_g = Super::_nu = Super::_delta = GP_Vector(Super::_d);

      // Initialize active and passive set
      Super::_I.clear();
      _J.clear();
      for(uint i=0; i<n; ++i) {
      	_J.push_back(i);
      }

      //_delta_values.resize(_J.size(), std::numeric_limits<double>::min());

      // initialize heap
      //_heap = HeapType(n, max_extvalue(_delta_values));

      // initialize covariance matrix
      _K_full = Super::ComputeCovarianceMatrix(Super::_hparms);

      std::cout << "J size: " << _J.size() << std::endl;
      std::cout << "_K_full: " << _K_full.Rows() << " x " << _K_full.Cols() << std::endl;

      for (size_t y = 0; y < _J.size(); ++y) {
      	std::cout << "Y = " << y << "/" << _J.size() << std::endl;
      	_delta_values[y] = computeMutualInformation(y);
      	_heap.push(y);
      }

      // Super::_d .. size of active set
      for(uint k=0; k<Super::_d; ++k){
				std::cout << "Iteration " << k << "/" << Super::_d << std::endl;

				// find next point for inclusion into the active set
				size_t argmax = _heap.top();
				_heap.pop();

				std::cout << "max MI = " << _delta_values[argmax] << std::endl;
				std::cout << "Y* = " << argmax << std::endl;

				// refine site params, posterior params and matrices M, L, and K
				Super::UpdateAll(argmax, Super::_g[k], Super::_nu[k], k, m, beta,
						 Super::_mu, Super::_zeta, Super::_M);

				// add idx to I and remove it from J
				Super::_I.push_back(argmax);
				_J.remove(argmax);

	      std::cout << "J size: " << _J.size() << std::endl;
	      std::cout << "I size: " << Super::_I.size() << std::endl;

	      std::vector<size_t> neighbours_psv = getNeighboursInPassiveSet(argmax);
	      for (int i=0; i<neighbours_psv.size(); ++i) {
	      	size_t y = neighbours_psv[i];
	      	std::cout << "i in N(Y*) = " << i << "/" << neighbours_psv.size() << ", Y = " << y << std::endl;
	      	_delta_values[y] = computeMutualInformation(y);
	      	_heap.update(y); //_heap.push(y);
	      }

     }

      // re-compute site parameters for numerical stability
      Super::ComputeSiteParams(m, beta, Super::_mu);

      // compute log posterior and  derivative wrt the kernel params
      Super::ComputeLogZ();
      Super::ComputeDerivLogZ();
    }

  protected:
    values_type _delta_values;
    Cmp _cmp;

    std::list<uint> _J;
    GP_Matrix _K_full, _Kpsv, _Lpsv;
    double _CovThresh;

    HeapType _heap;

    double computeMutualInformation(size_t y) {
    	InputType const &x = Super::GetX()[y];
    	std::vector<double>& cov_vec_y = _K_full[y];
      double sigma_Y = cov_vec_y[y]; //Super::CovFunc(x, x, Super::_hparms);

      //std::cout << "computeMutualInformation" << std::endl;

      //std::cout << "sigma_Y = " << sigma_Y << std::endl;
      //std::cout << "x = " << x << std::endl;

			double enumer = sigma_Y;

			// if we have an active set, we compute the mutual information
			// of y with respect to the active set and the passive set
			if(Super::_I.size() > 0){

				//std::cout << "got an active set already" << std::endl;

				// Compute k_star
				GP_Vector k_star_act; //(Super::_I.size());

				std::vector<size_t> neighbours_act;
				neighbours_act.reserve(Super::_I.size());
				size_t i = 0;
				for(std::list<uint>::const_iterator it = Super::_I.begin(); it != Super::_I.end(); ++it, ++i) {
					double cov_y_i = cov_vec_y[*it];
					if (cov_y_i > _CovThresh) {
						//std::cout << "cov_y_i " << cov_y_i << std::endl;
						k_star_act.Append(cov_y_i);
						neighbours_act.push_back(i);
					}
				}

				//std::cout << "k_star_act.Size() = " << k_star_act.Size() << std::endl;

				// compute (B^-1 + K)^-1 * k_star using our Cholesky decomp
				if (k_star_act.Size()>1) {
					//std::cout << neighbours_act.size() << "x" << neighbours_act.size() << std::endl;
					GP_Matrix L_act(neighbours_act.size(), neighbours_act.size());

		      for (size_t i=0; i<neighbours_act.size(); ++i) {
		        for (size_t j=0; j<neighbours_act.size(); ++j) {
		        	//std::cout << neighbours_act[i] << ", " << neighbours_act[j] << std::endl;
		        	L_act[i][j] = Super::_L[neighbours_act[i]][neighbours_act[j]];
		        }
		      }

					//std::cout << "SolveChol.." << std::endl;
					GP_Vector v = L_act.SolveChol(k_star_act);
					//std::cout << "enumer.." << std::endl;
					enumer = sigma_Y - k_star_act.Dot(v);
				} else {
					if (k_star_act.Size()>0) {
						enumer = sigma_Y - k_star_act[0];
					}
				}
				//std::cout << "enumer = " << enumer << std::endl;
			}

			//std::cout << "getting k_star passive" << std::endl;
			// Compute k_star passive
			GP_Vector k_star_psv; //(_J.size()-1);
			std::vector<size_t> neighbours_psv;
			neighbours_psv.reserve(_J.size()-1);
			for(std::list<uint>::const_iterator it = _J.begin(); it != _J.end(); ++it) {
				if (*it != y) {
					double cov_y_i = cov_vec_y[*it];
					if (cov_y_i > _CovThresh) {
						//std::cout << "cov_y_i " << cov_y_i << std::endl;
						k_star_psv.Append(cov_y_i);
						neighbours_psv.push_back(*it);
					}
				}
			}

			std::cout << "running estimation passive, neighbours_psv size: " << neighbours_psv.size() << std::endl;

			// compute Sigma of passive set
			EstimationPassiveSparse(k_star_psv, neighbours_psv);

			// compute square root of site variances
			GP_Vector s_sqrt = Super::MakeSqrtS();

			//std::cout << "computing posterior covariance passive" << std::endl;

			// compute posterior covariance of the passive set
			GP_Vector v = _Lpsv.ForwSubst(s_sqrt * k_star_psv);
			double denom = sigma_Y - v.Dot(v); //MAX(sigma_Y - v.Dot(v), 1e-5);

			// now we can compute the mutual information
			double mi = enumer / denom;

			std::cout << "mutual information " << mi << std::endl;

			return mi;
    }

    /*
   uint
    FindMaxMIPoint(GP_Vector const &mu, GP_Vector const &zeta, 
		   double &g_kmax, double &nu_kmax, double &MI_max)
    {

      std::cout << "=== FindMaxMIPoint ===" << std::endl;

      MI_max = -HUGE_VAL;
      std::list<uint>::iterator argmax = _J.begin();

      // loop through all passive points and see which one we can remove
      for (std::list<uint>::iterator candidate = _J.begin(); candidate != _J.end(); candidate++) {
				// we do not remove this point but skip it when computing the MI
				uint cand_idx = *candidate;

      	InputType const &x_star = Super::GetX()[cand_idx];
        double sigma_Y = Super::CovFunc(x_star, x_star, Super::_hparms);

        std::cout << "sigma_Y = " << sigma_Y << std::endl;
        std::cout << "next point to test: " << x_star << std::endl;

				double enumer = sigma_Y;

				std::cout << "\tnext candidate " << cand_idx << std::endl;
				std::cout << "J size " << _J.size() << std::endl;

				// if we don't have an active set, there is nothing to do here
				if(!Super::_I.empty()){

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
				GP_Vector s_sqrt = Super::MakeSqrtS();

				std::cout << "getting k_star passive" << std::endl;
				// Compute k_star passive
				GP_Vector k_star_psv(_J.size()-1);
				uint i = 0;
				for(std::list<uint>::const_iterator n = _J.begin();
						n != _J.end(); ++n, ++i) {
					if (*n < cand_idx) {
						k_star_psv[i] = Super::CovFunc(Super::GetX()[*n], x_star, Super::_hparms);
					} else if (*n > cand_idx) {
						k_star_psv[i-1] = Super::CovFunc(Super::GetX()[*n], x_star, Super::_hparms);
					}
				}

				std::cout << "computing posterior covariance passive" << std::endl;

				// compute posterior covariance of the passive set
				GP_Vector v = _Lpsv.ForwSubst(s_sqrt * k_star_psv);
				double denom = sigma_Y - v.Dot(v); //MAX(sigma_Y - v.Dot(v), 1e-5);

				// now we can compute the mutual information
				double mi = enumer / denom;

				std::cout << "got MI " << mi << std::endl;

				// and store the maximum
				if(mi > MI_max){
					MI_max = mi;
					argmax = candidate;

					std::cout << "found new max " << *argmax << ", " << MI_max << std::endl;
				}
      }      

      std::cout << "argmax is " << *argmax << std::endl;

      Super::ComputeDerivatives(Super::GetY()[*argmax], mu[*argmax] / zeta[*argmax],
				1./zeta[*argmax], g_kmax, nu_kmax);
      std::cout << "got derivs " << g_kmax << " " << nu_kmax << std::endl;

      uint rem_idx = *argmax;
			for(std::list<uint>::const_iterator it = Super::_I.begin();
				it != Super::_I.end(); ++it){
				if(*argmax > *it)
					--rem_idx;
			}

       std::cout << "RemoveRowAndColumn " << rem_idx << std::endl;
      _K.RemoveRowAndColumn(rem_idx);

      uint retval = *argmax;
      _J.erase(argmax);
      return retval;
    }
   */

    // computes Sigma of passive set
    void EstimationPassiveSparse(const GP_Vector& k_star_psv,	const std::vector<size_t>& neighbours_psv)
    {
      _Kpsv = GP_Matrix(neighbours_psv.size(), neighbours_psv.size());

      for (size_t i=0; i<neighbours_psv.size(); ++i) {
        for (size_t j=0; j<neighbours_psv.size(); ++j) {
        	_Kpsv[i][j] = _K_full[neighbours_psv[i]][neighbours_psv[j]];
        }
      }

      // initialize the site and posterior params
      uint n = neighbours_psv.size();
      Super::_nu_tilde = GP_Vector(n);
      Super::_tau_tilde = GP_Vector(n);

      GP_Vector mu(n);
      GP_Matrix Sigma = _Kpsv;

      //std::cout << "Sigma: " << Sigma.Rows() << " x " << Sigma.Cols() << std::endl;

      // now we run EP until convergence (or max number of iterations reached)
    	Super::_EPthresh = 1e-2;
      uint iter = 0;
      double max_delta = 1.;
      do {

				// run EP; the maximum difference in tau is our convergence criterion
				GP_Vector delta_tau = Super::ExpectationPropagation(mu, Sigma, neighbours_psv.begin(), neighbours_psv.end());
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
      GP_Vector s_sqrt = MakeSqrtSSparse();

      _Lpsv = _Kpsv;
      _Lpsv.CholeskyDecompB(s_sqrt);

      // Compute S^0.5 * K in place
      Sigma = GP_Matrix(n, n);
      for(uint i=0; i<n; i++)
				for(uint j=0; j<n; j++)
					Sigma[i][j] = s_sqrt[i] * _Kpsv[i][j];
      
      GP_Matrix V = _Lpsv.ForwSubst(Sigma);

      Sigma = _Kpsv - V.TranspTimes(V);
      mu    = Sigma * Super::_nu_tilde;
    }


    GP_Vector MakeSqrtSSparse() const
    {
      uint n = Super::_tau_tilde.Size();
      GP_Vector s_sqrt(n);

      for(uint i=0; i<n; i++)
				if(Super::_tau_tilde[i] > EPSILON)
					s_sqrt[i] = sqrt(Super::_tau_tilde[i]);
				else
					s_sqrt[i] = sqrt(EPSILON);

      return s_sqrt;
    }

    std::vector<size_t> getNeighboursInPassiveSet(size_t y)
    {
    	std::vector<double>& cov_vec_y = _K_full[y];
    	std::vector<size_t> neighbours_psv;
			neighbours_psv.reserve(_J.size());
			for(std::list<uint>::const_iterator it = _J.begin(); it != _J.end(); ++it) {
				if (*it != y) {
					double cov_y_i = cov_vec_y[*it];
					if (cov_y_i > _CovThresh) {
						neighbours_psv.push_back(*it);
					}
				}
			}

			return neighbours_psv;
    }

  };

}

#endif
