#ifndef KERNEL_PCA_HH
#define KERNEL_PCA_HH

#include "GPlib/GP_Base.hh"
#include "GPlib/GP_DataSet.hh"
#include "GPlib/GP_CovarianceFunction.hh"

namespace GPLIB {


  template<typename InputType, 
	   typename OutputType = uint,
	   typename KernelType = GP_SquaredExponential<InputType> >

  class KernelPCA : public GP_Base<InputType, OutputType, KernelType>
  {

  public:

    typedef GP_DataSet<InputType, OutputType> DataSet;
    typedef GP_Base<InputType, OutputType, KernelType> Super;
    typedef typename Super::HyperParameters HyperParameters;

    KernelPCA() {}

    KernelPCA(DataSet const &data) : Super(data) {}

    void Train(HyperParameters const &hparms)
    {
      uint n = Super::Size();

      _K = Super::ComputeCovarianceMatrix(hparms);
      _unit = GP_Matrix(n, n, 1. / n);

      GP_Matrix K_n  = _K - _unit * _K - _K * _unit + _unit * _K * _unit;
      GP_Vector evals;
      
      K_n.EigenSolveSymm(evals, _evecs);

      //std::cout << "K_n " << K_n << std::endl;
      //std::cout << "evecs " << _evecs << std::endl;

      for (uint i=0; i<n; ++i){
	for (uint j=0; j<n; ++j){
	  _evecs[j][i] = _evecs[j][i] / sqrt(evals[i]);
	}
      }
    }

    virtual double 
    LearnHyperParameters(std::vector<double> &hyp_init, 
			 GP_Vector lower_bounds = GP_Vector(1,0.0), 
			 GP_Vector upper_bounds = GP_Vector(1,1.0), 
			 uint nb_iterations = 0)
    {
      std::cout << "Warning! Learning of kernel parameters for "
		<< "kernel PCA not implemented." << std::endl;
      return 0;
    }


    GP_Matrix Predict(DataSet const &test_data, HyperParameters const &hparms, 
		      uint max_ev)
    {
      uint m = test_data.Size();
      uint n = Super::Size();
      GP_Matrix unit_test(m, n, 1. / n);
      GP_Matrix K_test = Super::ComputeCovarianceMatrix(test_data, 
							Super::GetTrainData(), 
							hparms);

      GP_Matrix K_test_n = K_test - unit_test * _K - K_test * _unit + unit_test* _K* _unit;

      GP_Matrix test_features(m, max_ev);
      for(uint i=0; i<max_ev; ++i)
	for(uint j=0; j<m; ++j)
	  test_features[j][i] = K_test_n.Row(j).Dot(_evecs.Col(i));

      return test_features;
    }

  private:
    
    GP_Matrix _K, _unit, _evecs;
  };

}


#endif
