#ifndef KERNELOSC_H
#define KERNELOSC_H


#include "GPlib/GP_CovarianceFunction.h"
#include "OnlineStarClustering.h"


namespace onlineclust {

  template<typename KernelType>
  class KernelOSC : public OnlineStarClustering  
  {
  public:
    KernelOSC(typename KernelType::HyperParameters hparms, double sigma = 0.7) : 
      OnlineStarClustering(sigma), _hparms(hparms) {}

    double computeSimilarity(Eigen::VectorXd const &x1,
			     Eigen::VectorXd const &x2) const
    {
      return KernelType()(x1, x2, _hparms);
    }

  private:

    typename KernelType::HyperParameters _hparms;
  };
}

#endif
