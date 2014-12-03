#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_DataReader.hh"
#include "GPlib/GP_PolyadicClassificationIVM.hh"

using namespace std;
using namespace GPLIB;


typedef std::vector<double> InputType;
typedef uint OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_PolyadicClassificationIVM<InputType> IVMClassifier;
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
  WRITE_FILE(info_file, "program_info_train.txt");

  // Read training and test data
  GP_DataReader<InputType, OutputType> *reader = 0;
  if(params.data_format == GP_InputParams::LASER_DATA){
    info_file << "loading laser data" << std::endl;
    reader = new GP_LaserDataReader<InputType, OutputType>(params);
  }
  else if(params.data_format == GP_InputParams::GTSRB_DATA){
    info_file << "loading GTSRB data" << std::endl;
    reader = new GP_GTSRBDataReader<InputType, OutputType>(params);
  }
  else if(params.data_format == GP_InputParams::TLR_DATA){
    info_file << "loading TLR data" << std::endl;
    reader = new GP_TLRDataReader<InputType, OutputType>(params);
  }
  DataSetType train_data = reader->Read(true);
  train_data.Shuffle();

  if(params.data_format != GP_InputParams::LASER_DATA)
    train_data.DownSample(1. - params.train_frac);
  
  train_data.Write("training_data2.dat");

  info_file << "training data size: " << train_data.Size() << std::endl;
  
  // Initialize IVM
  uint d = (uint) ceil(train_data.Size() * params.active_set_frac);
  info_file << "active set size: " << d << std::endl;

  std::vector<double> hparams = params.GetHyperParamsInit();

  info_file << "Instantiating IVM with kernel parameters ";
  for(uint j=0; j<hparams.size(); ++j)
    info_file << hparams[j] << " " << std::flush;
  info_file << std::endl;

  IVMClassifier classif_ivm(train_data, d, params.lambda, params.useEP, hparams);
    
  // Train the hyperparameters
  info_file << "training hyper parameters... " << std::endl;
  classif_ivm.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				   params.kparam_upper_bounds,
				   params.nb_iterations);

  WRITE_FILE(kfile, "kparams.dat");
  for(uint j=0; j<hparams.size(); ++j)
    kfile << hparams[j] << " " << std::flush;
  kfile << std::endl;

  /*
  // Write all interesting information
  classif_ivm.ExportActiveSet();
  classif_ivm.ExportLeastInformative();
  classif_ivm.ExportZeta();
  classif_ivm.Squeeze();
  classif_ivm.Write(ivm_filename);
    */
}

END_PROGRAM


