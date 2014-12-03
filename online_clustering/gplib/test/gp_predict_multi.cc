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

typedef GP_Vector InputType;
typedef uint OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_PolyadicClassificationIVM<InputType> IVMClassifier;
typedef IVMClassifier::HyperParameters HyperParameters;

#define GP_RANDOM_SEED_USEC 1

BEGIN_PROGRAM(argc, argv)
{
  if(argc != 3)
    throw GP_EXCEPTION2("Usage: %s <config_file> <model_file>", argv[0]);

  std::string ivm_filename = argv[2];

  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_clsf.txt");

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
  DataSetType test_data  = reader->Read(false);
  //DataSetType test_data  = reader->Read(true);
  

  info_file << "training data size: " << train_data.Size() << std::endl;
  info_file << "test data size: " << test_data.Size() << std::endl;

  // Initialize IVM
  uint d = (uint) ceil(train_data.Size() * params.active_set_frac);
  info_file << "active set size: " << d << std::endl;

  HyperParameters hparams;
  hparams.Read("kparams.dat");

  info_file << "Instantiating IVM with kernel parameters ";
  for(uint j=0; j<hparams.Size(); ++j)
    info_file << hparams.ToVector()[j] << " " << std::flush;
  info_file << std::endl;

  IVMClassifier classif_ivm(train_data, d, params.lambda, params.useEP, hparams.ToVector());
  classif_ivm.Estimation();
  classif_ivm.ExportActiveSet();
  classif_ivm.PlotClassif(0, 25, 0.2);

  uint m = classif_ivm.GetNbClasses();
  GP_Matrix conf_mat(m, m);

  // We run through the test data set and classify them
  WRITE_FILE(cfile, "classification.dat");
  std::vector<GP_Vector> zvals;
  std::vector<OutputType> yvals;
  
  double corr = 0, incorr = 0;
  for(uint test_idx = 0; test_idx < test_data.Size(); ++test_idx){
    
    InputType x   = test_data.GetInput()[test_idx];
    OutputType y  = test_data.GetOutput()[test_idx];

    GP_Vector mu_star;
    GP_Matrix sigma_star;
    GP_Vector z = classif_ivm.Prediction(x, mu_star, sigma_star);
    
    cfile << test_idx << " " << z << " " << z.ArgMax() << " " << y << std::endl;
    
    zvals.push_back(z);
    yvals.push_back(y);
    
    uint label = classif_ivm.GetLabel(z.ArgMax());

    if(label == y)
      corr++;
    else
      incorr++;

    conf_mat[label][y]++;
  }
  
  info_file << corr << " correct classifications" << std::endl;
  info_file << incorr << " incorrect classifications" << std::endl;
  info_file << corr / (corr + incorr) << " classification rate" << std::endl;
  info_file << conf_mat << std::endl;

  //GP_Evaluation::plot_pr_curve(zvals, yvals, "prec_rec.dat", 0.05);  
}

END_PROGRAM


