#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"
#include "GPlib/GP_Histogram.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_Regression.hh"

using namespace std;
using namespace GPLIB;

typedef GP_Vector InputType;
typedef double OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_Regression<InputType, OutputType>  Regression;


double Branin(double x1, double x2, double a = 1, 
	      double b = 5.1 / (4. * M_PI * M_PI), 
	      double c = 5. / M_PI, double r = 6., 
	      double s = 10., double t = 1/(8. * M_PI))
{
  double temp = (x2 - b*x1*x1 + c*x1 -r);

  return a * temp * temp + s * (1-t) * cos(x1) + s;
}

BEGIN_PROGRAM(argc, argv)
{
  WRITE_FILE(ofile, "bayes_opt.dat");

  for(double x1 = -5; x1 <= 10; x1 += 0.1){
    for(double x2 = 0; x2 <= 15; x2 += 0.1){
      ofile << x1 << " " << x2 << " " << Branin(x1, x2) << endl;
    }
    ofile << endl;
  }
  ofile.close();

  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  DataSetType training_data;
  std::vector<GP_Vector> x;
  std::vector<double> y;

  gsl_rng_env_setup();
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
      

  for(uint i=0; i<3; ++i){
    GP_Vector v(2);
    v[0] = gsl_rng_uniform(rng) * 15 - 5;
    v[1] = gsl_rng_uniform(rng) * 15;
    
    x.push_back(v);
    y.push_back(Branin(v[0], v[1]));

    std::cout << v << " " << y.back() << std::endl;
  }
  training_data.Append(x, y);

  std::vector<double> hparams = params.GetHyperParamsInit();
  Regression regr(training_data, hparams);

  std::cout << "training hyper parameters... " << std::endl;
  
  regr.LearnHyperParameters(hparams, params.kparam_lower_bounds,
			    params.kparam_upper_bounds,
			    params.nb_iterations);


  for(uint j=0; j<hparams.size(); ++j)
    std::cout << hparams[j] << " " << std::flush;
  std::cout << std::endl;



  gsl_rng_free(rng);

} END_PROGRAM
