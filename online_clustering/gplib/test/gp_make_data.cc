#include "GPlib/GP_Matrix.hh"
#include "GPlib/GP_DataSet.hh"
#include <stdlib.h>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"


using namespace GPLIB;


GP_DataSet<GP_Vector, int> make_gaussian_data(std::vector<GP_Vector> const &means,
					      std::vector<GP_Matrix> const &covs,
					      std::vector<uint> const &nb_samples)
{
  gsl_rng_env_setup();
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
  struct timeval tv;
  gettimeofday(&tv, 0);
  gsl_rng_set(rng, tv.tv_usec);    

  GP_DataSet<GP_Vector, int> data;
  
  for(uint i=0; i<means.size(); ++i){

    GP_Vector eval, smpl(means[i].Size());
    GP_Matrix evec;

    covs[i].EigenSolveSymm(eval, evec);

    std::cout << eval << std::endl;
    std::cout << evec << std::endl;

    std::vector<GP_Vector> samples;
    std::vector<int>   labels;

    std::cout << "making " << nb_samples[i]<< " samples" << std::endl;
    for(uint k=0; k<nb_samples[i]; ++k){
      for(uint j=0; j<means[i].Size(); ++j){
	smpl[j] = gsl_ran_gaussian(rng, sqrt(eval[j]));
      }

      samples.push_back(evec * smpl + means[i]);
      labels.push_back(i);
    }
    
    data.Append(samples, labels);
  }

  gsl_rng_free(rng);

  return data;
}



BEGIN_PROGRAM(argc, argv)
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  
  srandom(tv.tv_usec);

  uint K = 4;
  uint D = 2;
  
  std::vector<GP_Vector> means(K, GP_Vector(D));
  std::vector<GP_Matrix> covs(K, GP_Matrix(D,D));

  for(uint k=0; k<K; ++k){
    for(uint i=0; i<D; ++i)
      means[k][i] = random() / (double) RAND_MAX * 20.;
  }

  std::cout << " made means" << std::endl;

  for(uint k=0; k<K; ++k){
    for(uint i=0; i<D; ++i){
      for(uint j=0; j<D; ++j){
	covs[k][i][j] = 1 + random() / (double) RAND_MAX * 2.;
	if(i == j)
	  covs[k][i][j] += 1.;
      }
    }
    covs[k] = covs[k].TimesTransp(covs[k]);

    
  }

  std::vector<uint> lengths(K, 300);

  std::cout << " made covs" << std::endl;

  GP_DataSet<GP_Vector, int> data = 
    make_gaussian_data(means, covs, lengths);

  WRITE_FILE(gfile, "test_data.dat");
  for(uint k=0; k<K; ++k){
    for(uint i=0; i<data.Size(); i++){
      int y = data.GetOutput()[i];
      if(y == k){
	gfile << data.GetInput()[i]<< " \tclass" << y << std::endl;
      }
    }
    gfile << std::endl << std::endl;
  }
  gfile.close();

} END_PROGRAM
