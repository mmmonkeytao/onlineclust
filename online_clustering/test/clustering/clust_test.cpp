#include <iostream>
#include "GPlib/GP_CovarianceFunction.hh"
#include "Vertex.h"
#include "KernelOSC.h"
#include "Mesh.h"
#include "Cluster.h"
#include "OnlineStarClustering.h"

using namespace onlineclust;
using namespace std;

typedef GPLIB::GP_SquaredExponential<Eigen::VectorXd> KernelType;
typedef KernelType::HyperParameters HyperParameters;

int main(int argc, char **argv)
{
  // //HyperParameters hparms;
  // //hparms[1] = 0.3;

  if(argc != 5){
    cerr << "Usage: <./exec> <file dir> <sigma> <no. of data to be deleted> <beta(V_measure)>" << endl;
    exit(1);
  }

  double beta = atoi(argv[4]);
  //KernelOSC<KernelType> osc(hparms, atof(argv[2]));
  OnlineStarClustering osc(atof(argv[2]));

  osc.loadAndAddData(argv[1]);
  osc.exportDot("out_insert.dot", false);

  osc.V_measure(beta);

  list<uint> deleteList;
  uint flag[osc.getDataSize()];
  uint counter = 0;
  uint numtodelete = atoi(argv[3]);

  for(uint i = 0; i < osc.getDataSize(); i++) flag[i] = 0;

  while(counter < numtodelete){

    uint randn = rand() % osc.getDataSize(); 
    if(flag[randn] == 0){
      deleteList.push_back(randn); 
      flag[randn] = -1;
      counter++;
    }
  }
  
  osc.deleteData(deleteList);  
  osc.exportDot("out_delete.dot",false);   
  osc.V_measure(beta);

  return 0;
}
