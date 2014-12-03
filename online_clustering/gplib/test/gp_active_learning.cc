#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>


#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "GPlib/GP_DataSet.hh"
#include "GPlib/GP_Histogram.hh"

#include "GPlib/GP_InputParams.hh"
#include "GPlib/GP_Evaluation.hh"
#include "GPlib/GP_UniversalDataReader.hh"
#include "GPlib/GP_BinaryClassificationIVM.hh"

using namespace std;
using namespace GPLIB;

typedef GP_Vector InputType;
typedef int OutputType;
typedef GP_DataSet<InputType, OutputType> DataSetType;
typedef GP_BinaryClassificationEP<InputType>  GPClassifier;
typedef GP_BinaryClassificationIVM<InputType> IVMClassifier;
//typedef GP_BinaryClassificationIVM<InputType, GP_SquaredExponentialARD<InputType> > IVMClassifier;
typedef IVMClassifier::HyperParameters HyperParameters;

#define GP_RANDOM_SEED_USEC 1

class MoreUncertain
{
public:
  
  bool operator()(std::pair<uint64_t, double> const &a,
		  std::pair<uint64_t, double> const &b) const
  {
    return a.second > b.second;
  }
};

BEGIN_PROGRAM(argc, argv)
{
  if(argc != 2)
    throw GP_EXCEPTION2("Usage: %s <config_file>", argv[0]);

  gsl_rng_env_setup();

  // Read the program options from the config file
  GP_InputParams params;
  params.Read(std::string(argv[1]));
  params.Write("params.txt");

  // Open program log file
  WRITE_FILE(info_file, "program_info.txt");

  // Read training and test data
  GP_UniversalDataReader<InputType, OutputType> reader(params);
  DataSetType train_data = reader.Read(true);
  DataSetType test_data;

  // Down sample training data
  // if train and test are the same data file, down sample train and 
  // use the rest for testing
  if(params.test_file_name == params.train_file_name){
    std::cout << "using same file" << std::endl;
    test_data = train_data.DownSample(1. - params.train_frac);
  }
  else {
    train_data.DownSample(1. - params.train_frac);
    test_data = reader.Read(false);
  }
  test_data.Write("test_data.dat");
  train_data.Write("training_data.dat");

  DataSetType eval_data;
  if(params.eval_file_name != "none")
    eval_data = reader.Read(true);

  bool satisfied = false;
  uint test_idx = 0, epoch = 0, last_file_epochs = 0;
  std::vector<InputType> retrain_x;
  std::vector<OutputType> retrain_y;
  std::vector<pair<uint, double> > retrain_idx;

#ifdef GP_RANDOM_SEED_USEC
  std::vector<uint> test_map = DataSetType::MakeSamples(test_data.Size(), true);
#else
  std::vector<uint> test_map = DataSetType::MakeSamples(test_data.Size(), false);
#endif

  //HyperParameters last_hparams;
  std::vector<double> last_hparams;
  bool no_change = false;

  while(!satisfied){

#ifdef GP_RANDOM_SEED_USEC
    if(params.shuffle)
      train_data.Shuffle(true);
#else
    if(params.shuffle)
      train_data.Shuffle(false);
#endif

    info_file << "training data size: " << train_data.Size() << std::endl;
    info_file << "test data size: " << test_data.Size() << std::endl;

    // Initialize IVM
    uint d = (uint) ceil(train_data.Size() * params.active_set_frac);
    info_file << "active set size: " << d << std::endl;

    //HyperParameters hparams;
    std::vector<double> hparams;
    if(params.relearn != GP_InputParams::FULL && epoch != 0)
      hparams = last_hparams;
    else
      hparams = params.GetHyperParamsInit();

    info_file << "Instantiating IVM with kernel parameters ";
    for(uint j=0; j<hparams.size(); ++j)
      info_file << hparams[j] << " " << std::flush;
    info_file << std::endl;

    IVMClassifier classif_ivm(train_data, d, hparams, params.lambda,			    
			      params.opt_type, params.opt_params[0], 
			      params.opt_params[1], params.opt_params[2], 
			      params.useEP, params.verbose);
    
    // Train the hyperparameters
    std::stringstream kfname, cfname, efname;
    kfname << "kparams" << std::setw(3) << std::setfill('0') << epoch << ".dat";
    cfname << "classification" << std::setw(3) << std::setfill('0') << epoch << ".dat";
    if(params.eval_file_name != "none")
      efname << "evaluation" << std::setw(3) << std::setfill('0') << epoch << ".dat";

    WRITE_FILE(kfile, kfname.str().c_str());
    if(params.do_optimization && !no_change &&
       (params.relearn == GP_InputParams::FULL || epoch == 0)){
      info_file << "training hyper parameters... " << std::endl;

      time_t time1 = time(0);
      classif_ivm.LearnHyperParameters(hparams, params.kparam_lower_bounds,
      			       params.kparam_upper_bounds,
      			       params.nb_iterations);
      //classif_gp.LearnHyperParameters(hparams, params.kparam_lower_bounds,
      //			       params.kparam_upper_bounds,
      //			       params.nb_iterations);
      time_t time2 = time(0);
      std::cout << "time " << difftime(time2, time1);

      for(uint j=0; j<hparams.size(); ++j)
	kfile << hparams[j] << " " << std::flush;
      kfile << std::endl;
    }
    else
      classif_ivm.Estimation();
      //classif_gp.Estimation();
    last_hparams = classif_ivm.GetHyperParams().ToVector();
    //last_hparams = classif_gp.GetHyperParams();

    classif_ivm.ExportActiveSet();
    classif_ivm.ExportLeastInformative();
    classif_ivm.ExportZeta();
    
    // Now we run through the test data set and classify them
    WRITE_FILE(cfile, cfname.str().c_str());
    std::map<uint, uint> next_idx_to_val_idx;
    std::vector<double> zvals;
    std::vector<OutputType> yvals;
    uint cnt;

    double tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;

    double mu_max = 10.;
    double sigma_max = 10.;
    
    GP_Histogram mu_hist(10, make_pair(0, mu_max));
    GP_Histogram sigma_hist(10, make_pair(0., sigma_max));
    GP_Histogram ne_hist(10, make_pair(0.,1.));

    for(; test_idx < test_data.Size() && 
	  test_idx < (epoch + 1 - last_file_epochs) * params.batch_size; 
	++test_idx, ++cnt){

      uint next_idx = params.shuffle ? test_map[test_idx] : test_idx;
      InputType x   = test_data.GetInput()[next_idx];
      OutputType y  = test_data.GetOutput()[next_idx];

      double mu_star, sigma_star;
      double z = classif_ivm.Prediction(x, mu_star, sigma_star);
      //double z = classif_gp.Prediction(x);

      //double score = GP_Evaluation::calc_norm_entropy(z);
      double score;
      if(params.retraining_score == GP_InputParams::NE ||
	 params.retraining_score == GP_InputParams::RAND)
	score = GP_Evaluation::calc_norm_entropy(z);
      else if (params.retraining_score == GP_InputParams::BALD)
	score = classif_ivm.ComputeBALD(z, mu_star, sigma_star);
      else if  (params.retraining_score == GP_InputParams::EIG)
	score = classif_ivm.ComputeExpectedInformationGain(x);
      
      cfile << test_idx << " " << next_idx << " " << z << " " << y << " " 
	    << score << " " << mu_star << " " << sigma_star << std::endl;

      if(mu_star > -mu_max && mu_star < mu_max)
	mu_hist.Increment(fabs(mu_star));
      if(sigma_star < sigma_max)
	sigma_hist.Increment(sigma_star);
      ne_hist.Increment(score);

      next_idx_to_val_idx[next_idx] = zvals.size();
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

      if(score > params.max_entropy){ // too uncertain: retrain!
      	retrain_idx.push_back(make_pair(next_idx, score));
      }
    }


    ne_hist.Normalize();
    mu_hist.Normalize();
    sigma_hist.Normalize();

    std::stringstream hfname;
    hfname << "ne_hist" << std::setw(3) << std::setfill('0') << epoch << ".dat";
    ne_hist.WriteTextFile(hfname.str().c_str());
    hfname.str("");
    hfname << "mu_hist" << std::setw(3) << std::setfill('0') << epoch << ".dat";
    mu_hist.WriteTextFile(hfname.str().c_str());
    hfname.str("");
    hfname << "sigma_hist" << std::setw(3) << std::setfill('0') << epoch << ".dat";
    sigma_hist.WriteTextFile(hfname.str().c_str());


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
    info_file << retrain_idx.size() << " points above entropy threshold" << std::endl;

    cfname.str("");
    cfname << "prec_rec" << std::setw(3)  << std::setfill('0') << epoch << ".dat";
    GP_Evaluation::plot_pr_curve(zvals, yvals, cfname.str().c_str(), 0.05);

    cfname.str("");
    cfname << "retrain" << std::setw(3)  << std::setfill('0') << epoch << ".dat";
    WRITE_FILE(rfile, cfname.str().c_str());

    // There are two different strategies to select the points to be re-trained:
    // random sampling and using the normalized entropy
    if(params.retraining_score == GP_InputParams::RAND){
      info_file << "randomly selecting retraining points" << std::endl;

#ifdef GP_RANDOM_SEED_USEC
      std::vector<uint> retrain_random = 
	DataSetType::MakeSamples(params.batch_size, true);
#else 
      std::vector<uint> retrain_random = 
	DataSetType::MakeSamples(params.batch_size, false);
#endif

      for(uint i=0; i<params.nb_questions; ++i){

	uint idx;
	if(params.shuffle)
	  idx = test_map[retrain_random[i] + 
			 (epoch - last_file_epochs) * params.batch_size];
	else
	  idx = retrain_random[i] + (epoch - last_file_epochs) * params.batch_size;

	retrain_x.push_back(test_data.GetInput()[idx]);
	retrain_y.push_back(test_data.GetOutput()[idx]);

	rfile << idx << " " << test_data.GetOutput()[idx] << " " 
	      << zvals[next_idx_to_val_idx[idx]] << std::endl;
	//	      << zvals[idx - (epoch - last_file_epochs) * params.batch_size] << std::endl;
      }
    }
    else {

      sort(retrain_idx.begin(), retrain_idx.end(), MoreUncertain());

      for(uint i=0; i<MIN(params.nb_questions, retrain_idx.size()); ++i){
	uint idx = retrain_idx[i].first;

	retrain_x.push_back(test_data.GetInput()[idx]);
	retrain_y.push_back(test_data.GetOutput()[idx]);

	rfile << idx << " " << retrain_idx[i].second << " " 
	      << test_data.GetOutput()[idx] << " " 
	      << zvals[next_idx_to_val_idx[idx]] << std::endl;
	//	      << zvals[idx - (epoch - last_file_epochs) * params.batch_size] << std::endl;
      }
    }
    rfile.close();

    // Forgetting means removing points form the training set
    // Currently, there are three different methods to do that, where
    // MIN_DELTA_H seems to work best so far
    if(params.forget){
      if(train_data.Size() > params.max_train_data_size){
	
	uint nb_useless = train_data.Size() - params.max_train_data_size;
	info_file << "forgetting " << nb_useless << " training points" << std::endl;
	std::vector<uint> forget_points; 

	if(params.forget_mode == GP_InputParams::MIN_DELTA_H)
	  forget_points = classif_ivm.GetLeastInformativePoints(retrain_x.size());
	else if (params.forget_mode == GP_InputParams::MIN_VAR)
	  forget_points = classif_ivm.GetLeastVariantPoints(nb_useless);
	else
	  throw GP_EXCEPTION2("Unknown forgetting mode '%s'.", params.forget_mode);

	train_data.Remove(forget_points);
      }
    }

    // Add re-training points  to the training data, but only if there
    // is something to re-train, otherwise we don't run training again
    if(retrain_x.size() == 0 &&
       retrain_y.size() == 0)
      no_change = true;
    else {
      no_change = false;
      if(params.relearn != GP_InputParams::PASSIVE)
	train_data.Append(retrain_x, retrain_y); // 
    }

    retrain_x.clear();
    retrain_y.clear();
    retrain_idx.clear();
    
    ++epoch;

    if(test_idx >= test_data.Size()){
      test_idx = 0;
      last_file_epochs = epoch;
      info_file << "opening new test data file" << std::endl;
      test_data = reader.Read(false);
      if(test_data.Size() == 0)
	satisfied = true;
     else
	info_file << "opened new test data file" << std::endl;
    }

    // Evaluate the current classifier on a separate data set if given
    if(params.eval_file_name != "none"){
      WRITE_FILE(efile, efname.str().c_str());
      for(uint i=0; i < eval_data.Size(); ++i){

	InputType x   = eval_data.GetInput()[i];
	OutputType y  = eval_data.GetOutput()[i];
	double mu_star, sigma_star;
	double z = classif_ivm.Prediction(x, mu_star, sigma_star);
	//double z = classif_gp.Prediction(x);

	//double score = GP_Evaluation::calc_norm_entropy(z);
	double score;
	if(params.retraining_score == GP_InputParams::NE ||
	   params.retraining_score == GP_InputParams::RAND)
	  score = GP_Evaluation::calc_norm_entropy(z);
	else if (params.retraining_score == GP_InputParams::BALD)
	  score = classif_ivm.ComputeBALD(z, mu_star, sigma_star);
	else if  (params.retraining_score == GP_InputParams::EIG)
	  score = classif_ivm.ComputeExpectedInformationGain(x);
	efile << i << " " << z << " " << y << " " << score << std::endl;
      }
    }
  }
  
  info_file.close();
  //delete reader;  
}

END_PROGRAM


