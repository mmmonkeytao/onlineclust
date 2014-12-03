#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_PolyadicClassificationEP.hh"

using namespace std;
using namespace GPLIB;


typedef GP_Vector InputType;
typedef uint OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_PolyadicClassificationEP<InputType> EPClassifier;
typedef EPClassifier::HyperParameters HyperParameters;


BEGIN_PROGRAM(argc, argv)
{
  if(argc < 2)
    throw GP_EXCEPTION2("Usage: %s <config_file>", argv[0]);
  
  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_train.txt");

  // Read training and test data
  GP_UniversalDataReader<InputType, OutputType> reader(params);
  DataSetType all_data = reader.Read(true);
  DataSetType train_data, test_data;

  //train_data.Shuffle();
  /*
  if(params.test_file_name == params.train_file_name){
    std::cout << "using same file" << std::endl;
    test_data = train_data.DownSample(1. - params.train_frac);
  }
  else {
    train_data.DownSample(1. - params.train_frac);
    test_data = reader.Read(false);
    }*/
  
  std::vector<uint> idcs;
  for(uint i=0; i<200; ++i)
    idcs.push_back(i);
  train_data = all_data.GetSubset(idcs);
  idcs.clear();
  for(uint i=200; i<all_data.Size(); ++i)
    idcs.push_back(i);
  test_data = all_data.GetSubset(idcs);
  test_data.Write("test_data.dat");
  train_data.Write("training_data.dat");

  info_file << "training data size: " << train_data.Size() << std::endl;
  
  // Initialize EP
  uint d = (uint) ceil(train_data.Size() * params.active_set_frac);
  info_file << "active set size: " << d << std::endl;

  std::vector<double> hparams = params.GetHyperParamsInit();

  info_file << "Instantiating EP with kernel parameters ";
  for(uint j=0; j<hparams.size(); ++j)
    info_file << hparams[j] << " " << std::flush;
  info_file << std::endl;

  EPClassifier classif_ep(train_data, hparams, params.lambda);
  //EPClassifier classif_ep(train_data, hparams, params.lambda, params.opt_type, 
  //			  params.opt_params[0], params.opt_params[1], params.opt_params[2]);
  

  // Train the hyperparameters
  if(params.do_optimization){
    info_file << "training hyper parameters... " << std::endl;
    classif_ep.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				    params.kparam_upper_bounds,
				    params.nb_iterations);
  }
  else 
    classif_ep.Estimation();

  WRITE_FILE(kfile, "kparams.dat");
  for(uint j=0; j<hparams.size(); ++j)
    kfile << hparams[j] << " "  << std::flush;
  kfile << std::endl;


  // We run through the test data set and classify them
  WRITE_FILE(cfile, "classification.dat");
  std::vector<double> zvals;
  std::vector<OutputType> yvals;
  
  double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;
  for(uint test_idx = 0; test_idx < test_data.Size(); ++test_idx){
    
    InputType x   = test_data.GetInput()[test_idx];
    OutputType y  = test_data.GetOutput()[test_idx];

    GP_Vector mu_star;
    GP_Matrix sigma_star;
    GP_Vector zvec = classif_ep.Prediction(x, mu_star, sigma_star);
    
    cfile << test_idx << " " << zvec << " " << y << std::endl;

    if(zvec.ArgMax() == y)
      tp++;
    else
      fn++;

    /*zvals.push_back(z);
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
    neg++;*/

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
  
  //GP_Evaluation::plot_pr_curve(zvals, yvals, "prec_rec.dat", 0.05); 


}

END_PROGRAM


