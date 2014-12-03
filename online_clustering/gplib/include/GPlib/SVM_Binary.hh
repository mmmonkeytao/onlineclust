#ifndef SVM_BINARY_HH
#define SVM_BINARY_HH

#include "GPlib/GP_DataSet.hh"
#include "GPlib/GP_Exception.hh"
#include "GPlib/GP_CovarianceFunction.hh"
#include "svm.h"
#include <iomanip>

namespace GPLIB {

  template<typename InputType, 
	   template <typename> class KernelType = GP_SquaredExponential>
  class SVM_Binary 
  {

  public:

    typedef GP_DataSet<InputType, int> DataSetType;
    typedef typename KernelType<InputType>::HyperParameters HyperParameters;

    SVM_Binary(DataSetType const &train_data, HyperParameters const &hparms) : 
      its_train_data(train_data), its_hparms(hparms), 
      its_problem(), its_param(), its_model_ptr(0)
    {
      its_problem.y = 0;
      its_problem.x = 0;

      SetDefaultParams();      
      ConvertToSVMProblem();

      const char *errmsg = svm_check_parameter(&its_problem, &its_param);
      if(errmsg){
	std::cerr << "Error: " << errmsg << std::endl;
	throw GP_EXCEPTION("Could not run SVM.");
      }
    }


    virtual ~SVM_Binary()
    {
      if(its_problem.y != 0)
	delete [] its_problem.y;
      if(its_problem.x != 0){
	for(int i=0; i<its_problem.l; ++i)
	  if(its_problem.x[i] != 0)
	    delete its_problem.x[i];
	delete [] its_problem.x;
      }
      if(its_model_ptr != 0)
	svm_free_and_destroy_model(&its_model_ptr);
    }

    virtual bool IsA(char const *classname) const
    {
      return (std::string(classname) == "SVM_Binary");
    }

    HyperParameters const &GetHyperParams() const
    {
      return its_hparms;
    }

    virtual void
    Train()
    {

      std::cout << "SVM training" << std::endl;
      double best_c, best_g, fmeas_best = 0;
      double *target = new double[its_problem.l];


      for(double C = -3; C <= 6; C += 1){
	for(double g = -4; g <= 1; g += 1){
	  
	  its_param.gamma = exp(g);
	  its_param.C = exp(C);

	  svm_cross_validation(&its_problem, &its_param, 5, target);
	  
	  uint tp = 0, fp = 0, fn = 0, tn = 0;
	  for(int i=0; i<its_problem.l; ++i){
	    
	    if(target[i] == 1 && its_problem.x[i]->index == 1){
	      tp++;
	    }
	    else if(target[i] == -1 && its_problem.x[i]->index == -1){
	      tn++;
	    }
	    else if(target[i] == -1 && its_problem.x[i]->index == 1){
	      fn++;
	    }
	    else if(target[i] == 1 && its_problem.x[i]->index == -1){
	      fp++;
	    }
	  }

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

	  if(fmeas > fmeas_best){
	    fmeas_best = fmeas;
	    best_c = C;
	    best_g = g;
	  }
	}
      }
      
      its_param.C = exp(best_c);
      its_param.gamma = exp(best_g);

      its_model_ptr = svm_train(&its_problem, &its_param);
      
      delete [] target;
    }

    virtual double Predict(InputType const &test_input) const
    {
      if(its_model_ptr == 0)
	throw GP_EXCEPTION("No SVM model trained. Run Training first.");
      int nb_class = svm_get_nr_class(its_model_ptr);
      double *prob_est = new double [nb_class];
      int *labels = new int[nb_class];
      svm_get_labels(its_model_ptr, labels);

      uint dim = test_input.Size();
      struct svm_node *x = new struct svm_node[dim+1];
      for(uint i=0; i<dim; ++i){
	x[i].index = i;
	x[i].value = test_input[i];
      }
      x[dim].index = -1;

      double predict_label = svm_predict_probability(its_model_ptr, x, prob_est);

      double retval;
      for(int i=0; i<nb_class; ++i){
	if(labels[i] == 1)
	  retval = prob_est[i];
      }

      delete [] x;
      delete [] labels;
      delete [] prob_est;

      return retval;
    }

    int GetNbSupportVectors() const
    {
      if(its_model_ptr == 0)
	return 0;

      return its_model_ptr->l;
    }

    void ExportSupportVectors(char const *filename = "support_vectors") const
    {
      if(its_model_ptr == 0)
	return;

      static uint64_t idx = 0;

      std::cout << "exporting support vector set " << idx << std::endl;

      std::stringstream fname;
      fname << filename << std::setw(3) << std::setfill('0') << idx++ << ".dat";
      WRITE_FILE(ofile, fname.str().c_str());

      for(int i=0; i<its_model_ptr->l; ++i){
	int j=0;
	int idx = its_model_ptr->SV[i][j].index;
	while(idx >= 0){
	  if(idx == j)
	    ofile << its_model_ptr->SV[i][j].value << " ";
	  else
	    ofile << "0 ";
	  j++;
	  idx = its_model_ptr->SV[i][j].index;
	}
	//ofile << its_train_data.GetOutput()[its_model_ptr->sv_indices[i]] << std::endl;
	ofile << std::endl;
      }
      ofile.close();
    }



  private:

    DataSetType its_train_data;
    HyperParameters its_hparms;
    struct svm_problem its_problem;
    struct svm_parameter its_param;
    struct svm_model *its_model_ptr;

    void ConvertToSVMProblem()
    {
      uint size = its_train_data.Size();

      its_problem.l = size;
      its_problem.y = new double[size];
      its_problem.x = new struct svm_node* [size];

      uint dim = its_train_data.GetInputDim();

      for(uint i=0; i<size; ++i){
	its_problem.x[i] = new struct svm_node[dim+1];
	for(uint j=0; j<dim; ++j){
	  its_problem.x[i][j].index = j;
	  its_problem.x[i][j].value = its_train_data.GetInput(i)[j];
	}
	its_problem.x[i][dim].index = -1;
	its_problem.y[i] = its_train_data.GetOutput()[i];
      }
    }

    void SetDefaultParams()
    {
      its_param.svm_type = C_SVC;
      if(KernelType<InputType>::ClassName() == "GP_SquaredExponential")
	its_param.kernel_type = RBF;
      else
	throw GP_EXCEPTION("Unkown kernel type for SVM.");
      its_param.degree = 3;
      its_param.gamma = 1./its_train_data.GetInputDim();	// 1/k
      its_param.coef0 = 0;
      its_param.nu = 0.5;
      its_param.cache_size = 100;
      its_param.C = 1;
      its_param.eps = 1e-3;
      its_param.p = 0.1;
      its_param.shrinking = 1;
      its_param.probability = 1;
      its_param.nr_weight = 0;
      its_param.weight_label = NULL;
      its_param.weight = NULL;
    }
  };

}

#endif
