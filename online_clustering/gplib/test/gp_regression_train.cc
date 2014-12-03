#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_RegressionIVM.hh"

using namespace std;
using namespace GPLIB;


typedef GP_Vector InputType;
typedef double OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_DataSet<OutputType, InputType> DataSetTypeRev;
typedef GP_Regression<InputType, OutputType> Regressor;
//typedef GP_RegressionIVM<InputType, OutputType> Regressor;
typedef GP_Regression<OutputType, InputType> RegressorRev;
typedef GP_RegressionIVM<OutputType, InputType> RegressorRevSp;
typedef Regressor::HyperParameters HyperParameters;
typedef RegressorRev::HyperParameters HyperParametersRev;


BEGIN_PROGRAM(argc, argv)
{
  /*!
   * First we test the R^2 -> R mapping
   */

  if(argc != 2)
    throw "Usage: GP_regression <params_file>";

  WRITE_FILE(ofile, "train_data_regr.dat");

  vector<GP_Vector> train_x;
  vector<double> train_y;

 std:cout << "generating training data..." <<std::endl;

  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  
  for(uint i=0; i<30; ++i){
    for(uint j=0; j<30; ++j){
      GP_Vector xvec(2);
      xvec[0] = i; xvec[1] = j;
      
      train_x.push_back(xvec);
      train_y.push_back(sin(i/2.) * 1.3 * log(j*j*j + 1e-5));
      ofile << xvec << " " << train_y.back() << endl;
    }
    ofile << endl;
  }
  ofile.close();

  DataSetType data;
  data.Append(train_x, train_y);

  std::vector<double> hparams = params.GetHyperParamsInit();

  //uint d = (uint) ceil(data.Size() * params.active_set_frac);
  //std::cout << "d = " << d << std::endl;
  //Regressor regr(data, d, hparams);
  Regressor regr(data, hparams);

  std::cout << "initial estimation ..." << std::endl;

  regr.Estimation();
  std::cout << "log Z: " << regr.GetLogZ() << std::endl;
  std::cout << "deriv log Z: " << regr.GetDeriv() << std::endl;

  std::cout << "training hyper parameters... " << std::endl;
  
  time_t time1 = time(0);
  regr.LearnHyperParameters(hparams, params.kparam_lower_bounds,
			    params.kparam_upper_bounds,
			    params.nb_iterations);

  time_t time2 = time(0);
  std::cout << "time " << difftime(time2, time1);
  
  for(uint j=0; j<hparams.size(); ++j)
    std::cout << hparams[j] << " " << std::flush;
  std::cout << std::endl;

  gsl_rng_env_setup();


  std::cout << "estimation done" << std::endl;

  
  WRITE_FILE(ofile2, "test_data_regr.dat");
  for(double i=-10; i<40; i += 0.71){
    for(double j=-10; j<40; j += 1.23){
      GP_Vector test(2);
      test[0] = i;
      test[1] = j;
     
      double mean, var;
      regr.Prediction(test, mean, var);
      ofile2 << test << " " << mean << std::endl;    
    }
    ofile2 << endl;
  }

  ofile2.close();
  

  exit(1);

  /*!
   * Now we test the R -> R^3 mapping
   */

  WRITE_FILE(ofile3, "train_data2.dat");

  vector<GP_Vector> train_x2;
  vector<double> train_y2;

  uint k=0;
  for(double i=0; i<30; i += 1){
    GP_Vector yvec(3);
    yvec[0] = i;
    yvec[1] = 2. *sin(i); 
    yvec[2] = 2. * cos(i);
    train_x2.push_back(yvec);
    train_y2.push_back(k++);
    ofile3 << yvec << endl;
  }
  ofile3.close();

  DataSetTypeRev data_rev;
  data_rev.Append(train_y2, train_x2);

  HyperParametersRev hparams2;
  hparams2.sigma_f_sqr = 1;
  hparams2.length_scale = 1;
  hparams2.sigma_n_sqr = 0.01;
  RegressorRev regr2(data_rev, hparams2);
  RegressorRevSp regr2sp(data_rev, data_rev.Size() / 2, hparams2);


  std::cout << "estimating dense version" << std::endl;
  regr2.Estimation();

  std::cout << "estimating sparse version" << std::endl;
  regr2sp.Estimation();

  WRITE_FILE(ofile4, "test_data2.dat");
  for(double i=0; i<30; i += 0.1){
    GP_Vector mean;
    GP_Matrix var;
    regr2.Prediction(i, mean, var);
    ofile4 << mean << std::endl;    
  }

  ofile4.close();

  WRITE_FILE(ofile5, "test_data2sp.dat");
  for(double i=0; i<30; i += 0.1){
    GP_Vector mean;
    GP_Matrix var;
    regr2sp.Prediction(i, mean, var);
    ofile4 << mean << std::endl;    
  }

  ofile5.close();


}
END_PROGRAM
