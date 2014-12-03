#include <vector>
#include <iostream>
#include <fstream>


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
typedef GP_BinaryClassificationIVM<InputType> IVMClassifier;
typedef IVMClassifier::HyperParameters HyperParameters;


BEGIN_PROGRAM(argc, argv)
{
  GP_Matrix M(3,3);
  M[0][0] = 1.2; M[0][1] = 1.9; M[0][2] = 1.2;
  M[1][0] = 0.2; M[1][1] = 5.8; M[1][2] = 3.1;
  M[2][0] = -.5; M[2][1] = 4.4; M[2][2] = 8.9;

  GP_Matrix MM = M * M.Transp();

  std::cout << MM << std::endl;

  GP_Matrix C = MM;
  C.Cholesky();

  GP_Matrix MMinv = MM.InverseByChol();
  
  std::cout << MMinv * MM << std::endl;

  double dot  = C.Diag().Prod();

  std::cout << C << std::endl;
  std::cout << sqrt(MM.Det()) << " " << dot << std::endl;

  std::cout << "cc " << std::endl;
  std::cout << C.TranspTimes(C) << std::endl;

  std::cout << C.Det() << " " << dot << std::endl;

  GP_Vector eval;
  GP_Matrix evec;

  MM.EigenSolveSymm(eval, evec);

  std::cout <<  eval << evec << std::endl;

  std::cout << evec * GP_Matrix::Diag(eval) * evec.Transp() << std::endl;
  std::cout << evec.Transp() * GP_Matrix::Diag(eval) * evec << std::endl;

  exit(1);
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
  /*GP_DataReader<InputType, OutputType> *reader = 0;
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
  */

  GP_UniversalDataReader<InputType, OutputType> reader(params);

  DataSetType train_data = reader.Read(true);

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

  IVMClassifier classif_ivm(train_data, d,  hparams, params.lambda, 
			    params.opt_type, params.opt_params[0], 
			    params.opt_params[1], params.opt_params[2], 
			    params.useEP);
    
  // Train the hyperparameters
  if(params.do_optimization){
    info_file << "training hyper parameters... " << std::endl;
    classif_ivm.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				     params.kparam_upper_bounds,
				     params.nb_iterations);
  }
  else
    classif_ivm.Estimation();
  
  // Write all interesting information
  classif_ivm.ExportActiveSet();
  classif_ivm.ExportLeastInformative();
  classif_ivm.ExportZeta();
  classif_ivm.Squeeze();
  classif_ivm.Write(ivm_filename);
    
}

END_PROGRAM


