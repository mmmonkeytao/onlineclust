#include "Cluster.h"

using namespace std;
using namespace onlineclust;

/////////////////////////// CONSTRUCTORS //////////////////////////////////////////
Cluster:: Cluster()
{
  // Setting the defaults.
  _clusterSize = 0;
  _clusterCenterIndex = 0;   
}

Cluster:: Cluster(unsigned clusterCenterIndex, unsigned clusterSize)
{
  // Setting the defaults.
  _clusterCenterIndex = clusterCenterIndex;   
  _clusterSize = clusterSize;

}

/////////////////////////// DESTRUCTOR /////////////////////////////////////////
Cluster:: ~Cluster() 
{}

/////////////////////////// PUBLIC FUNCTIONS ////////////////////////////////////

// Cluster size
unsigned Cluster::getClusterSize() 
{
  return _clusterSize;
}

void Cluster::setClusterSize(unsigned clusterSize) 
{
  _clusterSize = clusterSize;
}


// Cluster center related.
unsigned Cluster::getClusterCenterIndex() 
{
  return _clusterCenterIndex;
}

void Cluster::setClusterCenterIndex(unsigned clusterCenterIndex) 
{
  _clusterCenterIndex = clusterCenterIndex;
}

void Cluster::printCluster()
{
  cout << "ClusterCenter: " << _clusterCenterIndex<<"\n";
  cout << "ClusterSize: " << _clusterSize<<"\n";
}
