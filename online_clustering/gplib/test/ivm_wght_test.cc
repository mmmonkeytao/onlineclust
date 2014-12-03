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



typedef std::vector<double> InputType1;
typedef std::pair<InputType1, double> InputType2;
typedef int OutputType;
typedef GP_DataSet<InputType1, OutputType> DataSetType1;
typedef GP_DataSet<InputType2, OutputType> DataSetType2;


typedef GP_BinaryClassificationIVM<InputType2, GP_SquaredExponentialWght> Classifier;
typedef GP_SquaredExponentialWght::HyperParameters HyperParameters;



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

  GP_UniversalDataReader<InputType1, OutputType> reader(params);

  DataSetType1 train_data = reader.Read(true);  
}
END_PROGRAM
