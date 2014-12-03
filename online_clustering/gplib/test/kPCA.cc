#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/KernelPCA.hh"

using namespace std;
using namespace GPLIB;


typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;


BEGIN_PROGRAM(argc, argv)
{
  if(argc < 2)
    throw GP_EXCEPTION2("Usage: %s <config_file> [model_file]", argv[0]);
  
  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info_kpca.txt");

  // Read training and test data
  GP_UniversalDataReader<InputType, OutputType> reader(params);

  DataSetType train_data = reader.Read(true);
  DataSetType test_data;

  if(params.test_file_name == params.train_file_name){
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

  train_data.Write("training_data_kpca.dat");
  test_data.Write("test_data_kpca.dat");

  info_file << "training data size: " << train_data.Size() << std::endl;
  info_file << "test data size: " << test_data.Size() << std::endl;

  KernelPCA<InputType, OutputType> kpca(train_data);

  kpca.Train(params.GetHyperParamsInit());

  info_file << "training done" << std::endl;

  if(test_data.Size() == 0){
    std::vector<GP_Vector> xvec;
    std::vector<int> yvec;
    for(double x=-1; x<= 1; x+= 1./7.){
      for(double y=-0.5; y<= 1.5; y+= 1./7.){
	GP_Vector vec(2);
	vec[0] = x;
	vec[1] = y;
	xvec.push_back(vec);
	yvec.push_back(1);
      }
    }
    test_data.Append(xvec, yvec);
  }

  GP_Matrix test = kpca.Predict(test_data, params.GetHyperParamsInit(), 10);

  info_file << "prediction done" << std::endl;

  std::cout << test << std::endl;

  for(uint d=0; d<10; ++d){
    std::stringstream fn;
    fn << "result_" << std::setfill('0') << std::setw(3) << d << ".dat";
    std::ofstream ofile (fn.str().c_str());

    for(uint i=0; i<test_data.Size(); ++i){
      ofile << test_data.GetInput()[i] << " " 
	    << test[i][d] << " " << test_data.GetOutput()[i] << std::endl;
      //if(i%15 == 14)
      //ofile  << std::endl;
    }
    ofile.close();
  }

} END_PROGRAM
