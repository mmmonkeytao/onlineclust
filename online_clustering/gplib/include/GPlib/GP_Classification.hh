#ifndef GP_CLASSIFICATION_HH
#define GP_CLASSIFICATION_HH

#include "GPlib/GP_Matrix.hh"
#include "GPlib/GP_DataSet.hh"
#include "GPlib/GP_SigmoidFunction.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_Base.hh"

namespace GPLIB {

  /*!
   * Base class for GP classification
   *
   * This class contains the sigmoid function used for classification 
   * and provides the interface for derived classification classes.
   */
  template<typename InputType, typename OutputType, 
	   typename SigmoidFuncType = GP_LogisticSigmoid,
	   typename Kernel = GP_SquaredExponential<InputType> >
  class GP_Classification : public GP_Base<InputType, OutputType, Kernel>
  {

  public:

    typedef Kernel KernelType;
    typedef GP_Base<InputType, OutputType, KernelType> Super;
    typedef typename KernelType::HyperParameters HyperParameters;
    
    /*!
     * Default constructor
     */
    GP_Classification() : 
      GP_Base<InputType, OutputType, KernelType>(),
      _sig_func()
    {}

    /*!
     * This constructor expects the trainig data set
     */
    GP_Classification(GP_DataSet<InputType, OutputType> const &train_data) : 
      GP_Base<InputType, OutputType, KernelType>(train_data),
      _sig_func()
    {}

    /*!
     * Copy constructor
     */
    GP_Classification(GP_Classification<InputType, OutputType, 
					SigmoidFuncType, KernelType> const &other) :
    GP_Base<InputType, OutputType, KernelType>(other),
    _sig_func(other._sig_func)
    {}

    /*!
     * Default destructor
     */
    virtual ~GP_Classification()
    {}

    /*!
     * Retruns true if the class name is correctly given
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      std::string(classname) == "GP_Classification");
    }

    /*!
     * Returns the sigmoid function
     */
    SigmoidFuncType const &GetSigFunc() const
    {
      return _sig_func;
    }

    /*!
     * Returns the result of the sigmoid function applied to a value 'x'
     */
    double SigFunc(double x) const
    {
      return _sig_func(x);
    }

    /*!
     * Returns the result of the derivative of the sigmoid function 
     * applied to a value 'x'
     */
    double SigFuncDeriv(double x) const
    {
      return _sig_func.Deriv(x);
    }

    double LogLikelihoodBin(double y, double f) const
    {
      return _sig_func.LogLikelihoodBin(y, f);
    }

    double LogLikelihoodBinDeriv(double y, double f) const
    {
      return _sig_func.LogLikelihoodBinDeriv(y, f);
    }

    double LogLikelihoodBin2ndDeriv(double y, double f) const
    {
      return _sig_func.LogLikelihoodBin2ndDeriv(y, f);
    }

    /*!
     * Returns \f$\int \sigma(x) \dot N(x | mean, cov) dx\f$ 
     */
    double IntegrateSigmoidWithGaussian(double mean, double cov) const
    {
      return _sig_func.IntegrateWithGaussian(mean, cov);
    }

    int Read(std::string filename, int pos = 0)
    {
      int new_pos = Super::Read(filename, pos);

      std::string sfname;
      READ_FILE(ifile, filename.c_str());
      ifile.seekg(new_pos);
      ifile >> sfname;

      if(sfname != SigmoidFuncType::ClassName())
	throw GP_EXCEPTION2("Could not load model file: "
			    "incorrect sigmoid function type %s", sfname);

      return ifile.tellg();
    }

    void Write(std::string filename) const
    {
      Super::Write(filename);
      APPEND_FILE(ofile, filename.c_str());
      ofile << SigmoidFuncType::ClassName() << std::endl;
    }

  private:

    SigmoidFuncType _sig_func;

  };

}

#endif
