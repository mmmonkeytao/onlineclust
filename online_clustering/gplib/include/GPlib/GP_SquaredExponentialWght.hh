#ifndef SQUARED_EXPONENTIAL_WGHT_HH
#define SQUARED_EXPONENTIAL_WGHT_HH


#include "GPlib/GP_CovarianceFunction.hh"

namespace GPLIB {

class GP_WeightedVector : public GP_Vector
{
public:

  GP_WeightedVector() : 
    GP_Vector(), _weight(0)
  {}

  GP_WeightedVector(GP_Vector const &v, double w = 1) : 
    GP_Vector(v), _weight(w)
  {}

  double GetWeight() const 
  {
    return _weight;
  }

private:
  
  double _weight;

};


class GP_SquaredExponentialWght : public GP_SquaredExponential<GP_WeightedVector>
{
public:
  
  typedef GP_SquaredExponential<GP_WeightedVector> Super;
  typedef Super::HyperParameters HyperParameters;
  
  static std::string ClassName()
  {
    return "GP_SquaredExponentialWght";
  }
  
  double operator()(GP_WeightedVector const &arg1, GP_WeightedVector const &arg2,
		    HyperParameters const &hparams) const
  {
    double sqr_dist = Super::SqrDistance(arg1, arg2);
    double retval = hparams.sigma_f_sqr * arg1.GetWeight() * arg2.GetWeight() *
      exp(-sqr_dist / (2.0 * SQR(hparams.length_scale)));
    
    // add noise variance to diagonal
    if(sqr_dist < 1e-12) 
      retval += hparams.sigma_n_sqr;
    
    return retval;
  }

    // computes k(arg, arg), i.e. the diagonal of the covariance matrix
    double operator()(GP_WeightedVector const &arg,
		      HyperParameters const &hparams) const
    {
      return hparams.sigma_f_sqr * arg.GetWeight() * arg.GetWeight() + hparams.sigma_n_sqr;
    }

    double PartialDerivative(GP_WeightedVector const &arg1, GP_WeightedVector const &arg2,
			     HyperParameters const &hparams, uint idx) const
    {
      double sqr_dist =  Super::SqrDistance(arg1, arg2); 
      double sqr_length = SQR(hparams.length_scale);

      // derivative with respect to sigma_f_sqr
      if(idx == 0)
	return arg1.GetWeight() * arg2.GetWeight() * exp(-sqr_dist/(2.0 * sqr_length));

      // derivative with respect to the length scale
      else if(idx == 1)
	return hparams.sigma_f_sqr * arg1.GetWeight() * arg2.GetWeight() * 
	  sqr_dist / (hparams.length_scale * sqr_length) * 
	  exp(-sqr_dist / (2.0 * sqr_length));
      
      // derivative with respect to the noise variance
      else 
	return (sqr_dist < 1e-12);
    }

    double PartialDerivative(GP_WeightedVector const &arg, 
			     HyperParameters const &hparams, uint idx) const
    {
      // derivative with respect to sigma_f_sqr
      if(idx == 0) 
	return arg.GetWeight() * arg.GetWeight();

      // derivative with respect to the length scale
      else if(idx == 1)
	return 0;
      
      // derivative with respect to the noise variance
      else 
	return 1.;
    }

    std::vector<double> Derivative(GP_WeightedVector const &arg1, GP_WeightedVector const &arg2,
				   HyperParameters const &hparams) const
    {
      std::vector<double> deriv(hparams.Size());
      
      for(uint i=0; i<deriv.size(); ++i)
	deriv[i] = PartialDerivative(arg1, arg2, hparams, i);
    
      return deriv;
    }
  
  private:
  
  };


}

#endif
