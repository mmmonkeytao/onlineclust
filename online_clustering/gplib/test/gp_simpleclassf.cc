#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_BinaryClassificationIVM.hh"

using namespace std;
using namespace GPLIB;


typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_BinaryClassificationEP<InputType> EPClassifier;
typedef EPClassifier::HyperParameters HyperParameters;

class Less
{
public:
  bool operator()(std::pair<double, int> const &p1,
		  std::pair<double, int> const &p2) const
  {
    return p1.first < p2.first;
  }
};

void sort(  std::vector<std::vector<double> > &xvec,
	    std::vector<int> &yvec)
{
  std::vector<pair<double, int> > svec;
  for(uint i=0; i<xvec.size(); ++i)
    svec.push_back(make_pair(xvec[i][0], yvec[i]));
  
  std::sort(svec.begin(), svec.end(), Less());

  for(uint i=0; i<svec.size(); ++i){
    xvec[i][0] = svec[i].first;
    yvec[i] = svec[i].second;
  }
}

BEGIN_PROGRAM(argc, argv)
{
  gsl_rng_env_setup();
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
  struct timeval tv;
  gettimeofday(&tv, 0);
  gsl_rng_set(rng, tv.tv_usec); 

  // make some data
  std::vector<std::vector<double> > xvec;
  std::vector<int> yvec;
  uint N = 100;
  for(uint i=0; i<N; ++i){

    double smpl = (double)gsl_rng_uniform_int(rng, RAND_MAX) / RAND_MAX;
    xvec.push_back(std::vector<double>(1, smpl));

    if(smpl < 0.2 || smpl > 0.6)
      yvec.push_back(-1);
    else
      yvec.push_back(1);
  }

  sort(xvec, yvec);

  DataSetType train_data;
  train_data.Append(xvec, yvec);
  train_data.Write("train_data_1d.dat");

  HyperParameters hparams;
  if(argc >=  4){
    hparams.sigma_f_sqr = atof(argv[1]);
    hparams.length_scale = atof(argv[2]);
    hparams.sigma_n_sqr = atof(argv[3]);
  }
  else {
    hparams.sigma_f_sqr = 1;
    hparams.length_scale = .1;
    hparams.sigma_n_sqr = .001;
  }

  EPClassifier classif(train_data, hparams.ToVector());
  classif.Estimation();

  WRITE_FILE(lmfile, "latent_mu.dat");
  GP_Vector mu = classif.GetPosteriorMean();
  GP_Matrix Sigma = classif.GetPosteriorCov();

  for(uint i=0; i<mu.Size(); ++i){
    lmfile << xvec[i][0] << " " << mu[i] << " " << sqrt(Sigma[i][i]) << std::endl;
  }
  lmfile.close();

  WRITE_FILE(smfile, "sigmoid_mu.dat");
  for(uint i=0; i<mu.Size(); ++i){
    smfile << xvec[i][0] << " " << GP_CumulativeGaussian()(mu[i]) << std::endl;
  }
  smfile.close();

} END_PROGRAM
