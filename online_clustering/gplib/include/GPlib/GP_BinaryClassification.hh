#ifndef GP_BINARY_CLASSIFICATION_HH
#define GP_BINARY_CLASSIFICATION_HH

#include "GPlib/GP_Classification.hh"

namespace GPLIB {

  /*!
   * \class GP_BinaryClassification
   *
   * Base class for GP binary classification
   *
   * This is a special case of classification, where the output is 
   * either 1 for class or -1 for non-class, i.e. the OutputType is 'int'
   */
  template<typename InputType, 
	   typename SigmoidFuncType = GP_LogisticSigmoid,
	   typename KernelType = GP_SquaredExponential<InputType> >
  class GP_BinaryClassification : 
    public GP_Classification<InputType, int, SigmoidFuncType, KernelType>
  {

  public:

    typedef GP_Classification<InputType, int, 
			      SigmoidFuncType, KernelType> Super;
    typedef typename Super::HyperParameters  HyperParameters;
    typedef GP_DataSet<InputType, int>      DataSet;
    
    /*!
     * Default constructor
     */
    GP_BinaryClassification() : 
      Super()
    {}

    /*!
     * This constructor expects the trainig data set
     */
    GP_BinaryClassification(DataSet const &train_data) : 
      Super(train_data)
    {}

    GP_BinaryClassification(GP_BinaryClassification<InputType, SigmoidFuncType, 
						    KernelType> const &other) : 
      Super(other)
    {}

    /*!
     * Default destructor
     */
    virtual ~GP_BinaryClassification()
    {}

    /*!
     * Returns true if the class name is correctly given
     */
    virtual bool IsA(char const *classname) const
    {
      return (Super::IsA(classname) ||
	      std::string(classname) == "GP_BinaryClassification");
    }

    /*!
     * This function performs the prediction once the training is done. It expects an input 
     * test point of the correct type and returns the probability that 'test_input' has
     * class label 1.
     */
    double Prediction(InputType const &test_input) const
    {
      double mu_star, sigma_star;
      return Prediction(test_input, mu_star, sigma_star);
    }

    /*!
     * This is the same as 'Prediction(test_input)', with the difference that here
     * the predictive mean and variance are returned, too. Must be implemented by 
     * derived classes.
     */
    virtual double Prediction(InputType const &test_input, 
			      double &mu_star, double &sigma_star) const = 0;

  };

}

#endif
