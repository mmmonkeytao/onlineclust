#include <vector>
#include <iostream>
#include <fstream>

#define USE_BOOST_SERIALIZATION 1

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>

#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_PolyadicClassificationEP.hh"

using namespace std;
using namespace GPLIB;

typedef std::vector<double> InputType;
typedef uint OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_PolyadicClassificationEP<InputType> Classifier;
typedef Classifier::HyperParameters HyperParameters;


BEGIN_PROGRAM(argc, argv)
{
  if(argc < 2)
    throw GP_EXCEPTION2("Usage: %s <config_file> ", argv[0]);
  
  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_train.txt");

  // Read training and test data
  GP_UniversalDataReader<InputType, OutputType> reader(params);
  DataSetType train_data = reader.Read(true);
  DataSetType test_data;

  train_data.Write("training_data_m1.dat");

  if(params.test_file_name == params.train_file_name){
    std::cout << "using same file" << std::endl;
    test_data = train_data.DownSample(1. - params.train_frac);
  }
  else {
    train_data.DownSample(1. - params.train_frac);
    test_data = reader.Read(false);
  }

  train_data.Write("training_data_m2.dat");

  info_file << "training data size: " << train_data.Size() 
	    << ", input dimension: " << train_data.GetInputDim() << std::endl;
  
  // Initialiaze hyperparameters for standard kernel:
  std::vector<double> hparams = params.GetHyperParamsInit();


  // Initialize EP
  info_file << "Instantiating EP with kernel parameters ";
  for(uint j=0; j<hparams.size(); ++j)
    info_file << hparams[j] << " " << std::flush;
  info_file << std::endl;

  Classifier classif_ep(train_data, hparams, params.lambda);


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
    kfile << hparams[j] << " " << std::flush;
  kfile << std::endl;

  
  uint m = classif_ep.GetNbClasses();
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
    GP_Vector z = classif_ep.Prediction(x, mu_star, sigma_star);
    
    cfile << test_idx << " " << z << " " << z.ArgMax() << " " << y << std::endl;
    
    zvals.push_back(z);
    yvals.push_back(y);
    
    uint label = classif_ep.GetLabel(z.ArgMax());

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
}

END_PROGRAM


