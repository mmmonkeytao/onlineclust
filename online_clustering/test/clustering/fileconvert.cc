#include <iostream>
#include <fstream>
#include <ostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <list>
#include <iterator>

#define BRICKFACE 0.0
#define SKY       1.0
#define FOLIAGE   2.0
#define CEMENT    3.0
#define WINDOW 	  4.0
#define PATH      5.0
#define GRASS     6.0

using namespace std;

static const string LABELS[7] = {"BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"};

list< list<double> > normalize( list< list<double> > data );
list< list<double> > read_input_file(ifstream &ifile);
void write_output_file(ofstream &ofile, list< list<double> > data_set);

int main(int argc, char **argv){

    if(argc != 3)
	{
	  cerr << "Usage: <./exec> <input dir> <output dir>" << endl;
	  exit(1);
	}

	ifstream ifile(argv[1]);
	ofstream ofile(argv[2]);

	list<double> vec;
	list< list<double> > data_set = read_input_file(ifile);

	data_set = normalize( data_set );

        write_output_file(ofile, data_set);

        ifile.close();
	ofile.close();
}

list< list<double> > read_input_file(ifstream &ifile)
{
  list< list<double> > dataset;

  string line;
  size_t pos;

  while( getline(ifile, line) )
    {
      pos = 0;
      size_t next_pos;
      list<double> ls;

	while(pos < line.size()){

	    next_pos = line.find_first_of(',', pos);

	    if(next_pos == string::npos) next_pos = line.size() + 1;

            string substr = line.substr(pos, next_pos - pos);

	    double val;

	    if(!substr.compare(LABELS[0])){
	    	val = BRICKFACE;
	    } else if(!substr.compare(LABELS[1])){
	    	val = SKY;
	    } else if(!substr.compare(LABELS[2])){
	    	val = FOLIAGE;
	    } else if(!substr.compare(LABELS[3])){
	    	val = CEMENT;
	    } else if(!substr.compare(LABELS[4])){
	    	val = WINDOW;
	    } else if(!substr.compare(LABELS[5])){
	    	val = PATH;
	    } else if(!substr.compare(LABELS[6])){
	    	val = GRASS;
	    } else {
	      val = atof(substr.c_str());
	    }

	    ls.push_back(val);

	    pos = next_pos + 1;
	}

	dataset.push_back(ls);
    }

  return dataset;

}

list< list<double> > normalize( list< list<double> > data ){
  
	int i, j;
	double sum = 0.0;
	double vec[19];
	for(i = 0; i < 19; i++) vec[i] = 0.0;

	//compute average
	for(list< list<double> >::iterator it = data.begin(); it!=data.end(); ++it){
	  j = 0;
	  for(list<double>::iterator it1 = ++((*it).begin()); it1 != (*it).end(); ++it1){
	    vec[j] += *it1;
	    j++;
	  }  
	}
		 
	for(i = 0; i < 19; i++) vec[i] /= (double)data.size();

	for(list< list<double> >::iterator it = data.begin(); it!=data.end(); ++it){
	  j = 0;
	  for(list<double>::iterator it1 = ++((*it).begin()); it1 != (*it).end(); ++it1){
	    *it1 -= vec[j];
	    j++;
	  }  
	}

	// normalization
	for(list< list<double> >::iterator it = data.begin(); it!=data.end(); ++it){
	  j = 0;
	  for(list<double>::iterator it1 = ++((*it).begin()); it1 != (*it).end(); ++it1){
	                sum += (double)(*it1) * (double)(*it1);
			sum = sqrt(sum);
			j++;
	  }

	  j = 0;
	  for(list<double>::iterator it1 = ++((*it).begin()); it1 != (*it).end(); ++it1){
	                (*it1) /= (double)sum;
			j++;
	  }	  

	}
	
	return data;
 }


void write_output_file(ofstream &ofile, list< list<double> > data_set)
{

  uint rest_num = data_set.size();
  uint idx;
  
  // radomly store
  while(data_set.size() > 0){
      idx = rand() % rest_num;
      list<double> ls = *next(data_set.begin(), idx);

      for(list<double>::iterator it = ls.begin(); it != ls.end(); ++it) ofile << *it << ' ';
 
      ofile << endl;
      
      data_set.erase( next(data_set.begin(), idx) );

      rest_num--;
  }
}
