#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_BinaryClassificationMI.hh"

using namespace std;
using namespace GPLIB;


typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_BinaryClassificationIVM<InputType> IVMClassifier;
typedef IVMClassifier::HyperParameters HyperParameters;


BEGIN_PROGRAM(argc, argv)
{
  if(argc < 2)
    throw GP_EXCEPTION2("Usage: %s <config_file> [model_file]", argv[0]);
  
  std::string ivm_filename;
  if(argc == 2)
    ivm_filename = "ivm_model.dat";
  else
    ivm_filename = argv[2];

  gsl_rng_env_setup();

  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_ivm.txt");

  // Read training and test data
  GP_UniversalDataReader<InputType, OutputType> reader(params);
  std::vector<uint> idcs;
  for(uint i=0; i<30; ++i)
    idcs.push_back(i);

  DataSetType train_data = reader.Read(true);
  DataSetType train_data1 = train_data.GetSubset(idcs);

  for(uint i=30; i<40; ++i)
    idcs.push_back(i);
  DataSetType train_data2 = train_data.GetSubset(idcs);

  for(uint i=40; i<50; ++i)
    idcs.push_back(i);
  DataSetType train_data3 = train_data.GetSubset(idcs);

  idcs.clear();
  for(uint i=30; i<40; ++i)
    idcs.push_back(i);
  DataSetType ndata1 = train_data.GetSubset(idcs);

  idcs.clear();
  for(uint i=40; i<50; ++i)
    idcs.push_back(i);
  DataSetType ndata2 = train_data.GetSubset(idcs);

  train_data1.Write("training_data1.dat");
  train_data2.Write("training_data2.dat");


  struct timeval tv1, tv2;
  uint nb_iter = 20000;

  // Initialize IVM
  uint d = (uint) ceil(train_data1.Size() * params.active_set_frac);
  info_file << "active set size: " << d << std::endl;

  uint d2 = (uint) ceil(train_data2.Size() * params.active_set_frac);
  info_file << "active set size: " << d2 << std::endl;

  uint d3 = (uint) ceil(train_data3.Size() * params.active_set_frac);
  info_file << "active set size: " << d3 << std::endl;

  std::vector<double> hparams = params.GetHyperParamsInit();

  std::cout << "Instantiating IVM with kernel parameters ";
  for(uint j=0; j<hparams.size(); ++j)
    std::cout << hparams[j] << " " << std::flush;
  std::cout << std::endl;


  double sum1 = 0, sum2 = 0;
  double start_time, end_time, time_diff;
  for(uint i=0; i<nb_iter; ++i){
    IVMClassifier classif_ivm1(train_data1, d, hparams, params.lambda, 
			       params.opt_type, params.opt_params[0], 
			       params.opt_params[1], params.opt_params[2], 
			       params.useEP, false);
    
    classif_ivm1.Estimation();

    InputType test = train_data.GetInput()[60];
    double mu_star, sigma_star, mu_star1, sigma_star1;
    GP_Vector v0, m0;
    double z = classif_ivm1.PredictionIncremental(test, mu_star, sigma_star, v0, m0);
    std::cout << "res1 : " << z << " " << mu_star << " " << sigma_star << std::endl;

    classif_ivm1.AddTrainingData(ndata1);
    
    std::cout << "d " << d << " " << d2 << " " << d3 << std::endl;

    gettimeofday(&tv1, 0);
    start_time = tv1.tv_sec + tv1.tv_usec / 1000000.;
    classif_ivm1.EstimationIncremental(d2 - d);
    gettimeofday(&tv2, 0);
    end_time = tv2.tv_sec + tv2.tv_usec / 1000000.;
    time_diff = (end_time - start_time);
    sum1 += time_diff;
    
    std::cout << "pred normal" << std::endl;
    z = classif_ivm1.Prediction(test, mu_star1, sigma_star1);
    std::cout << "res2a : " << z << " " << mu_star1 << " " << sigma_star1 << std::endl;

    z = classif_ivm1.PredictionIncremental(test, mu_star, sigma_star, v0, m0);
    std::cout << "res2inca : " << z << " " << mu_star << " " << sigma_star << std::endl << std::endl;

    classif_ivm1.AddTrainingData(ndata2);
    classif_ivm1.EstimationIncremental(d3 - d2);
    z = classif_ivm1.Prediction(test, mu_star1, sigma_star1);
    std::cout << "res2b : " << z << " " << mu_star1 << " " << sigma_star1 << std::endl;
    
    z = classif_ivm1.PredictionIncremental(test, mu_star, sigma_star, v0, m0);
    std::cout << "res2incb : " << z << " " << mu_star << " " << sigma_star << std::endl << std::endl;



  }


  for(uint i=0; i<nb_iter; ++i){
    IVMClassifier classif_ivm2(train_data2, d2, hparams, params.lambda, 
			       params.opt_type, params.opt_params[0], 
			       params.opt_params[1], params.opt_params[2], 
			       params.useEP, false);
    
    gettimeofday(&tv1, 0);
    start_time = tv1.tv_sec + tv1.tv_usec / 1000000.;

    classif_ivm2.Estimation();

    gettimeofday(&tv2, 0);
    end_time = tv2.tv_sec + tv2.tv_usec / 1000000.;
    time_diff = (end_time - start_time);
    sum2 += time_diff;
  }
    
  std::cout << "time1 " << sum1 << std::endl;
  std::cout << "time2 " << sum2 << std::endl;

}END_PROGRAM


