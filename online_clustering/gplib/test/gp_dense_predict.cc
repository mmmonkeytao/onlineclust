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


void Read(EPClassifier& classif_ep, std::string filename) {
    // make an archive
    std::ifstream ifs;
    ifs.open(filename.c_str(), ios::binary);
    ArchiveTypeIn ia(ifs);
    // read class state from archive
    ia >> classif_ep;
}

#define GP_RANDOM_SEED_USEC 1

BEGIN_PROGRAM(argc, argv)
{
  if(argc != 3)
    throw GP_EXCEPTION2("Usage: %s <config_file> <model_file>", argv[0]);

  std::string gp_model_filename = argv[2];

  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_clsf.txt");

  // Read training and test data
  // Read training and test data

  GP_UniversalDataReader<InputType, OutputType> reader(params);

  std::cout << "reading test data" << std::endl;

  DataSetType test_data;
	test_data = reader.Read(false);

  info_file << "test data size: " << test_data.Size() << std::endl;

  // Make an empty classifier
  EPClassifier classif_ep;
  std::cout << "reading" << std::endl;
  Read(classif_ep, gp_model_filename);

  std::cout << "clsf" << std::endl;
  // We run through the test data set and classify them
  WRITE_FILE(cfile, "classification.dat");
  std::vector<double> zvals;
  std::vector<OutputType> yvals;
  
  double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;
  for(uint test_idx = 0; test_idx < test_data.Size(); ++test_idx){
    
    InputType x   = test_data.GetInput()[test_idx];
    OutputType y  = test_data.GetOutput()[test_idx];

    //std::cout << x << " " << y << std::endl;

    double mu_star, sigma_star;
    double z = classif_ep.Prediction(x, mu_star, sigma_star);
    
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


