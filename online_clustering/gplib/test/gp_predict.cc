#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_DataReader.hh"
#include "GPlib/GP_BinaryClassificationIVM.hh"

using namespace std;
using namespace GPLIB;

typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_BinaryClassificationIVM<InputType> IVMClassifier;
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
  DataSetType test_data  = reader->Read(false);

  info_file << "test data size: " << test_data.Size() << std::endl;

  // Make an empty classifier
  IVMClassifier classif_ivm;
  std::cout << "reding" << std::endl;
  classif_ivm.Read(ivm_filename);
  std::cout << "writing" << std::endl;
  classif_ivm.Write("test_ivm.dat");

  std::cout << "clsf" << std::endl;
  // We run through the test data set and classify them
  WRITE_FILE(cfile, "classification.dat");
  std::vector<double> zvals;
  std::vector<OutputType> yvals;
  
  double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;
  for(uint test_idx = 0; test_idx < test_data.Size(); ++test_idx){
    
    InputType x   = test_data.GetInput()[test_idx];
    OutputType y  = test_data.GetOutput()[test_idx];

    std::cout << x << " " << y << std::endl;

    double mu_star, sigma_star;
    double z = classif_ivm.Prediction(x, mu_star, sigma_star);
    
    cfile << test_idx << " " << z << " " << y << std::endl;
    
    zvals.push_back(z);
    yvals.push_back(y);
    
    if(z >= 0.5 && y == 1){
      // true positive
      tp++;
    }
    else if(z >= 0.5 && y == -1){
      // false positive
      fp++;
    }
    else if(z < 0.5 && y == 1){
      // false negative
      fn++;
    }
    else if(z < 0.5 && y == -1){
      // true negative
      tn++;
    }
    
    if(y == 1)
      pos++;
    else
      neg++;
  }
  
  // evaluate the classification
  double prec, rec, fmeas;
  if(tp + fp == 0)
    prec = 1.0;
  else
    prec = tp / (tp + fp);
  
  if(tp + fn == 0)
    rec = 1.0;
  else
    rec = tp / (tp + fn);
  
  fmeas = (1 + SQR(0.5)) * prec * rec / (SQR(0.5) * prec + rec);
  
  double corr = tp + tn;
  double incorr = fn + fp;
  
  info_file << pos << " " << neg  << " pos neg" << std::endl;
  info_file << tp << " " << fp << " " << fn << " " << tn << " tp fp fn tn" << std::endl;
  info_file << prec << " " << rec << " " << fmeas << " prec rec fmeas" << std::endl;
  info_file << corr << " correct classifications" << std::endl;
  info_file << incorr << " incorrect classifications" << std::endl;
  info_file << corr / (corr + incorr) << " classification rate" << std::endl;
  
  GP_Evaluation::plot_pr_curve(zvals, yvals, "prec_rec.dat", 0.05);  
}

END_PROGRAM


