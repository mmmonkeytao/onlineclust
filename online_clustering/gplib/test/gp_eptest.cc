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
//typedef GP_BinaryClassificationEP<InputType> EPClassifier;
typedef GP_BinaryClassificationEP<InputType, GP_SquaredExponentialARD<InputType> > EPClassifier;
typedef EPClassifier::HyperParameters HyperParameters;

uint wid = 450;
uint hght = 600;

void NormalizeImage(DataSetType &test_set)
{
  if(test_set.GetInputDim() == 5){
    for(uint i=0; i<test_set.Size(); ++i){
      GP_Vector vec = test_set.GetInput(i);
      vec[0] /= wid;
      vec[1] /= hght;
      vec[2] /= 255.;
      vec[3] /= 255.;
      vec[4] /= 255.;
      test_set.SetInput(i, vec);
    }
  }
}

BEGIN_PROGRAM(argc, argv)
{
  if(argc < 2)
    throw GP_EXCEPTION2("Usage: %s <config_file> [model_file]", argv[0]);
  
  std::string ep_filename;
  if(argc == 2)
    ep_filename = "ep_model.dat";
  else
    ep_filename = argv[2];

  gsl_rng_env_setup();

  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_train.txt");

  // Read training and test data

  GP_UniversalDataReader<InputType, OutputType> reader(params);

  DataSetType train_data = reader.Read(true);

  std::cout << "reading test data" << std::endl;

  DataSetType test_data, clsf_data;
  if(params.test_file_name == params.train_file_name){
    std::cout << "using same file" << std::endl;
    //test_data = train_data.DownSample(1. - params.train_frac);
    uint n = (uint)(train_data.Size() * params.train_frac);
    vector<uint> idcs_train(n), idcs_test(train_data.Size() - n);
    for(uint i=0; i<train_data.Size(); ++i)
      if(i < n)
	idcs_train[i] = i;
      else
	idcs_test[i-n] = i;

    test_data = train_data.GetSubset(idcs_test);
    train_data = train_data.GetSubset(idcs_train);
  }
  else {
    train_data.DownSample(1. - params.train_frac);
    test_data = reader.Read(false);
  }

  train_data.Write("training_data_ep.dat");
  //test_data.WritePNM("test_img.pnm", wid, hght);

  info_file << "training data size: " << train_data.Size() << std::endl;
  info_file << "test data size: " << test_data.Size() << std::endl;

  //NormalizeImage(train_data);
  //NormalizeImage(test_data);

  test_data.Write("test_data_norm.dat");
  
  // Initialize EP
  std::vector<double> hparams = params.GetHyperParamsInit();

  info_file << "Instantiating EP with kernel parameters ";
  for(uint j=0; j<hparams.size(); ++j)
    info_file << hparams[j] << " " << std::flush;
  info_file << std::endl;


  EPClassifier classif_ep(train_data, hparams, params.lambda, params.opt_type,
  			  params.opt_params[0], params.opt_params[1], params.opt_params[2]);
  
  // Train the hyperparameters
  if(params.do_optimization){
    info_file << "training hyper parameters... " << std::endl;
    classif_ep.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				    params.kparam_upper_bounds,
				    params.nb_iterations);
  }
  else{
    classif_ep.Estimation();
    classif_ep.PreparePrediction();
  }
  
  std::cout << "logZ " << classif_ep.GetLogZ() << std::endl;
  std::cout << "deriv " << classif_ep.GetDeriv() << std::endl;

  // We run through the test data set and classify them
  clsf_data = test_data;
  WRITE_FILE(cfile, "classification.dat");
  std::vector<double> zvals;
  std::vector<OutputType> yvals;
  
  double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;
  for(uint test_idx = 0; test_idx < test_data.Size(); ++test_idx){
    
    InputType x   = test_data.GetInput()[test_idx];
    OutputType y  = test_data.GetOutput()[test_idx];

    double mu_star, sigma_star;
    double z = classif_ep.Prediction(x, mu_star, sigma_star);
    
    clsf_data.SetInput(test_idx, x * 255.);
    clsf_data.SetOutput(test_idx, z > 0.5 ? 1 : -1);

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

  clsf_data.WritePNM("clsf_img.pnm", wid, hght);

  info_file << pos << " " << neg  << " pos neg" << std::endl;
  info_file << tp << " " << fp << " " << fn << " " << tn << " tp fp fn tn" << std::endl;
  info_file << prec << " " << rec << " " << fmeas << " prec rec fmeas" << std::endl;
  info_file << corr << " correct classifications" << std::endl;
  info_file << incorr << " incorrect classifications" << std::endl;
  info_file << corr / (corr + incorr) << " classification rate" << std::endl;
  
  GP_Evaluation::plot_pr_curve(zvals, yvals, "prec_rec.dat", 0.05);  
  

}

END_PROGRAM


