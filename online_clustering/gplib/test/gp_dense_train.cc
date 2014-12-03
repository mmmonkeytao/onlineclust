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
#include "GPlib/GP_BinaryClassificationEP.hh"

using namespace std;
using namespace GPLIB;

typedef std::vector<double> InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;

//template <typename T> class KernelType : public GP_SquaredExponential<T> {};
template <typename T> class KernelType : public GP_SquaredExponentialARD<T> {};
typedef GP_BinaryClassificationEP<InputType, KernelType> EPClassifier;
typedef KernelType<InputType>::HyperParameters HyperParameters;

// use the default CovarianceFunction GP_SquaredExponential:
//typedef GP_BinaryClassificationEP<InputType> EPClassifier;
//typedef EPClassifier::KernelType KernelType;
//typedef KernelType::HyperParameters HyperParameters;

/*
 * Different file formats for serialization
 */
// a portable text archive
//typedef boost::archive::text_oarchive ArchiveTypeOut; // saving
//typedef boost::archive::text_iarchive ArchiveTypeIn; // loading

// a portable text archive using a wide character stream
//typedef boost::archive::text_woarchive ArchiveTypeOut; // saving
//typedef boost::archive::text_wiarchive ArchiveTypeIn; // loading

// a portable XML archive
//typedef boost::archive::xml_oarchive ArchiveTypeOut; // saving
//typedef boost::archive::xml_iarchive ArchiveTypeIn; // loading

// a portable XML archive which uses wide characters - use for utf-8 output
//typedef boost::archive::xml_woarchive ArchiveTypeOut; // saving
//typedef boost::archive::xml_wiarchive ArchiveTypeIn; // loading

// a non-portable native binary archive
typedef boost::archive::binary_oarchive ArchiveTypeOut; // saving
typedef boost::archive::binary_iarchive ArchiveTypeIn; // loading


void Save(const EPClassifier& classif_ep, std::string filename) {
    // make an archive
    std::ofstream ofs;
    ofs.open(filename.c_str(), ios::binary);
    ArchiveTypeOut oa(ofs);
    oa << classif_ep;
}

void initializeBounds(std::vector<double>& lower_bounds, std::vector<double>& upper_bounds, size_t length_scale_dim) {
  std::vector<double> lower_bounds_tmp;
  std::vector<double> upper_bounds_tmp;

  lower_bounds_tmp.push_back(lower_bounds[0]);
  for (int i=0; i<length_scale_dim; ++i) {
    lower_bounds_tmp.push_back(lower_bounds[1]);
  }
  if(lower_bounds.size() > 2)
    lower_bounds_tmp.push_back(lower_bounds[2]);

  upper_bounds_tmp.push_back(upper_bounds[0]);
  for (int i=0; i<length_scale_dim; ++i) {
    upper_bounds_tmp.push_back(upper_bounds[1]);
  }
  if(upper_bounds.size() > 2)
    upper_bounds_tmp.push_back(upper_bounds[2]);

  lower_bounds = lower_bounds_tmp;
  upper_bounds = upper_bounds_tmp;
}


BEGIN_PROGRAM(argc, argv)
{
  if(argc < 2)
    throw GP_EXCEPTION2("Usage: %s <config_file> [model_file]", argv[0]);
  
  std::string ivm_filename;
  if(argc == 2)
    ivm_filename = "gp_dense_model.dat";
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

  GP_UniversalDataReader<InputType, OutputType> reader(params);

  DataSetType train_data = reader.Read(true);

  train_data.DownSample(1. - params.train_frac);
  train_data.Write("training_data_dense_train.dat");

  info_file << "training data size: " << train_data.Size() 
	    << ", input dimension: " << train_data.GetInputDim() << std::endl;
  
  // Initialiaze hyperparameters and bounds for ARD-kernel:
  HyperParameters hparams(params.GetHyperParamsInit(train_data.GetInputDim()));
  initializeBounds(params.kparam_lower_bounds, params.kparam_upper_bounds, train_data.GetInputDim());

  // Initialiaze hyperparameters for standard kernel:
  //HyperParameters hparams(params.GetHyperParamsInit());

  // Initialize EP
  info_file << "Instantiating EP with kernel parameters ";
  for(uint j=0; j<hparams.Size(); ++j)
    info_file << hparams.ToVector()[j] << " " << std::flush;
  info_file << std::endl;

  EPClassifier classif_ep(train_data, hparams, params.lambda);


  // Train the hyperparameters
  if(params.do_optimization){
    info_file << "training hyper parameters... " << std::endl;
    classif_ep.LearnHyperParameters(hparams, params.kparam_lower_bounds,
				    params.kparam_upper_bounds,
				    params.nb_iterations);
  }
  else
    classif_ep.Estimation();
  
  // Write all interesting information
  Save(classif_ep, ivm_filename);
}

END_PROGRAM


