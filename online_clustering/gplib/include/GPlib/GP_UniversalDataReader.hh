#ifndef GP_UNIVERSAL_DATA_READER_HH
#define GP_UNIVERSAL_DATA_READER_HH

#include <fstream>
#include "GPlib/GP_DataReader.hh"


namespace GPLIB {

  template<typename InputType, typename OutputType>
  class GP_UniversalDataReader : public GP_DataReader<InputType, OutputType>
  {
  public:

    typedef GP_DataReader<InputType, OutputType> Super;

    GP_UniversalDataReader(GP_InputParams const &params) :
      GP_DataReader<InputType, OutputType>(params) {}

    virtual GP_DataSet<InputType, OutputType> 
    Read(bool train = true) const
    {
      std::vector<InputType> xvec;
      std::vector<OutputType> yvec;

      GP_InputParams params = Super::GetParams();
      std::string filename;
      static bool test_file_done = false;
      if(train) {
	filename = params.train_file_name;
      }
      else {
	filename = params.test_file_name;
	if(test_file_done)
	  return GP_DataSet<InputType, OutputType>();
      }

      std::cout << "loading " << filename << std::endl;

      std::string fext;
      size_t d = filename.find_last_of('.');
      if(d != std::string::npos) 
	fext = filename.substr(d+1);
      else
	throw GP_EXCEPTION("Could not load data file. File name has no file extension.");
      
      READ_FILE(ifile, filename.c_str());

      if(fext == "nfv") { // non-indexed feature vectors
	do {
	  std::vector<double> vec = GP_InputParams::ReadVector(ifile);

	  if(!ifile) break;
	  OutputType out = vec.back();
	  vec.pop_back();
	  xvec.push_back(InputType(vec));
	  yvec.push_back(out);

	} while(1);
      }
      else if(fext == "jf") { // non-indexed feature vectors
	uint nb_classes, nb_dims;
	ifile >> nb_classes >> nb_dims;
	std::cout << nb_classes << " " << nb_dims << std::endl;
	do {
	  uint label;
	  ifile >> label;

	  std::vector<double> vec = GP_InputParams::ReadVector(ifile);
	  if(!ifile) break;

	  if(label == params.label1){
	    xvec.push_back(InputType(vec));
	    yvec.push_back(OutputType(-1));
	  }
	  else if (label == params.label2){
	    xvec.push_back(InputType(vec));
	    yvec.push_back(OutputType(1));
	  }

	} while(1);
      }
      else if (fext == "dat") {

	std::map<std::string, uint> class2id;
	uint nb_classes = 0;
	uint nb_feat = 0;
	std::string token, labelstr;
	do{
	  
	  double val;
	  std::vector<double> vec;
	  while(ifile >> token && (token[0] == '#' || sscanf(token.c_str(), "%lf", &val) == 1)){
	    if(token[0] == '#'){
	      std::string line;
	      std::getline(ifile, line);
	    }
	    else 
	      vec.push_back(val);
	  }
	  if(!ifile) break;

	  if(nb_feat == 0)
	    nb_feat = vec.size();
	  else if(nb_feat != vec.size())
	    throw GP_EXCEPTION("Invalid length of feature vector");

	  labelstr = token;

	  if(class2id.find(labelstr) == class2id.end())
	    class2id[labelstr] = nb_classes++;

	  if(class2id[labelstr] == params.label1){
	    xvec.push_back(InputType(vec));
	    yvec.push_back(OutputType(-1));
	  }
	  else if (class2id[labelstr] == params.label2){
	    xvec.push_back(InputType(vec));
	    yvec.push_back(OutputType(1)); 
	  }
	} while(1);
      }
      else if (fext == "reg") {

	std::cout << "right" << std::endl;

	uint nb_feat = 0;
	std::string token;
	do{
	  
	  double val;
	  std::vector<double> vec;
	  while(ifile >> token && (token[0] == '#' || sscanf(token.c_str(), "%lf\n", &val) == 1)){

	    std::cout << "token " << token << std::endl;
	    if(token[0] == '#'){
	      std::string line;
	      std::getline(ifile, line);
	    }
	    else 
	      vec.push_back(val);
	  }
	  if(!ifile) break;

	  std::cout << vec.size() << std::endl;

	  if(nb_feat == 0)
	    nb_feat = vec.size() - 1;
	  else if(nb_feat != vec.size() - 1)
	    throw GP_EXCEPTION("Invalid length of feature vector");

	  val = vec.back();
	  vec.pop_back();
	  xvec.push_back(InputType(vec));
	  yvec.push_back(OutputType(val));

	} while(1);
      }
      else if (fext == "mult") {

	std::map<std::string, uint> class2id;
	uint nb_classes = 0;
	uint nb_feat = 0;
	std::string token, labelstr;
	std::map<uint, uint> label_map;
	std::cout << "label map: " << params.label_map.size() << std::endl;
	for(uint i=0; i<params.label_map.size(); ++i)
	  label_map[params.label_map[i].first] = params.label_map[i].second;

	std::cout << "label map: " << label_map.size() << std::endl;
	do{
	  
	  double val;
	  std::vector<double> vec;
	  while(ifile >> token && (token[0] == '#' || sscanf(token.c_str(), "%lf", &val) == 1)){
	    if(token[0] == '#'){
	      std::string line;
	      std::getline(ifile, line);
	    }
	    else 
	      vec.push_back(val);
	  }
	  if(!ifile) break;

	  if(nb_feat == 0)
	    nb_feat = vec.size();
	  else if(nb_feat != vec.size())
	    throw GP_EXCEPTION("Invalid length of feature vector");

	  labelstr = token;

	  if(class2id.find(labelstr) == class2id.end())
	    class2id[labelstr] = nb_classes++;

	  uint y = class2id[labelstr];
	  std::map<uint, uint>::const_iterator lmit = label_map.find(y);
	  if(lmit != label_map.end())
	    y = lmit->second;

	  xvec.push_back(InputType(vec));
	  yvec.push_back(OutputType(y));
	  
	} while(1);
      }
      else if (fext == "ifv") {

	OutputType y;
	uint max_len = 0;

	std::vector<std::vector<double> > vals_vec;
	uint min_idx = -1, max_idx = 0;
	while(ifile >> y) {

	  // we read the whole line and parse each element
	  std::vector<double> vec;
	  std::string line;
	  std::getline(ifile, line);
	  
	  size_t pos = 0;
	  std::vector<double> vals;
	  uint non_zero_idx, curr_idx = 0;
	  double val;
	  
	  do {
	    std::string substr;
	    size_t next_pos = line.find_first_of(' ', pos+1);
	    substr = line.substr(pos, next_pos - pos);
	    
	    sscanf(substr.c_str(), "%d:%lf", &non_zero_idx, &val);

	    min_idx = MIN(non_zero_idx, min_idx);
	    max_idx = MAX(non_zero_idx, max_idx);

	    if(non_zero_idx >= curr_idx)
	      vals.resize(non_zero_idx + 1);
	    vals[non_zero_idx] = val;
	    pos = next_pos;
	    
	  } while(pos < line.size());
	  
	  max_len = MAX(vals.size(), max_len);
	  if(max_len > vals.size())
	    vals.resize(max_len);

	  vals_vec.push_back(vals);
	  if(y == params.label1)
	    yvec.push_back(OutputType(-1));
	  else if(y == params.label2)
	    yvec.push_back(OutputType(1));
	}
	uint veclen = max_idx - min_idx + 1;
	std::cout << "veclen " << veclen << std::endl;

	for(uint i=0; i<vals_vec.size(); ++i){
	  InputType x(veclen);
	  for(uint j=min_idx; j < vals_vec[i].size(); ++j)
	    x[j-min_idx] = vals_vec[i][j];
	  xvec.push_back(x);
	}
      }

      if(!train)
	test_file_done = true;
      
      std::cout << "loaded " << xvec.size() << " feature vectors." << std::endl;
      GP_DataSet<InputType, OutputType> data;
      data.Append(xvec, yvec);

      return data;
    }

  private:
    
  };

}


#endif
