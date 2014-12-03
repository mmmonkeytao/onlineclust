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
#include "GPlib/GP_SquaredExponentialWght.hh"


using namespace std;
using namespace GPLIB;



typedef int OutputType;
typedef GP_DataSet<GP_Vector, OutputType> DataSetType1;
typedef GP_DataSet<GP_WeightedVector, OutputType> DataSetType2;


typedef GP_BinaryClassificationEP<GP_WeightedVector, 
				  GP_SquaredExponentialWght> EPClassifier;
typedef GP_SquaredExponentialWght::HyperParameters HyperParameters;

void WriteWghtData(std::string const &filename, DataSetType2 const &data)
{
  WRITE_FILE(file, filename.c_str());
  for(uint i=0; i<data.Size(); ++i){
    GP_WeightedVector input = data.GetInput()[i];
    for(uint j=0; j<input.Size(); ++j)
      file << input[j] << " ";
    file << input.GetWeight() << " " << data.GetOutput()[i] << std::endl;
  }
  file.close();
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

  GP_UniversalDataReader<GP_Vector, OutputType> reader(params);

  DataSetType1 train_data = reader.Read(true);
  //train_data.DownSample(1. - params.train_frac);

  params.label_map.clear();
  GP_UniversalDataReader<GP_Vector, OutputType> reader2(params);
  DataSetType1 train_data2 = reader2.Read(true);
  
  std::vector<GP_Vector> input1 = train_data.GetInput();
  std::vector<int> output1 = train_data.GetOutput();
  std::vector<int> output2 = train_data2.GetOutput();
  std::vector<GP_WeightedVector> input_wght;
  std::vector<int> output_wght;
  bool is_binary = train_data.Has(-1);

  for(uint i=0; i<input1.size(); ++i){

    double weight;
    
    if(is_binary)
      weight = 1;

    else if(output1[i] == params.label1){
      output_wght.push_back(-1);    
      if(output2[i] == params.label1)
	weight = 0.9999;
      else
	weight = 0.0001;
    }
    else if(output1[i] == params.label2){
      output_wght.push_back(1);    
      if(output2[i] == params.label2)
	weight = 0.9999;
      else
	weight = 0.0001;    
    }

    weight = 1;

    input_wght.push_back(GP_WeightedVector(input1[i], weight));    
  }

  DataSetType2 train_data_wght;
  if(is_binary)
    train_data_wght.Append(input_wght, output1);
  else
    train_data_wght.Append(input_wght, output_wght);
  train_data_wght.DownSample(1. - params.train_frac);

  train_data.Write("training_data_ep.dat");
  WriteWghtData("training_data_ep_wght.dat", train_data_wght);

  info_file << "training data size: " << train_data.Size() << std::endl;
  
  // Initialize EP
  std::vector<double> hparams = params.GetHyperParamsInit();

  info_file << "Instantiating EP with kernel parameters ";
  for(uint j=0; j<hparams.size(); ++j)
    info_file << hparams[j] << " " << std::flush;
  info_file << std::endl;

  EPClassifier classif_ep(train_data_wght, hparams, params.lambda);
  
  std::cout << "training" << std::endl;

  // Train the hyperparameters
  if(params.do_optimization){
    info_file << "training hyper parameters... " << std::endl;
    classif_ep.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				    params.kparam_upper_bounds,
				    params.nb_iterations);
  }
  else
    classif_ep.Estimation();
  
  // We run through the test data set and classify them
  WRITE_FILE(cfile, "classification.dat");
  std::vector<double> zvals;
  std::vector<OutputType> yvals;
  
  double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;

  std::vector<double> x_test(2);
  for(double xval = -15; xval <= 25; xval += 0.5){
    for(double yval = -15; yval <= 25; yval += 0.5){
    
	x_test[0] =  xval;	x_test[1] =  yval;
	
	GP_WeightedVector xw(x_test, 1);
	double mu_star, sigma_star;
	double z = classif_ep.Prediction(xw, mu_star, sigma_star);
    
	cfile << x_test << " " << z << std::endl;
    
	zvals.push_back(z);
      }
    
    cfile << std::endl;
  }
}

END_PROGRAM


