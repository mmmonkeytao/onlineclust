#include "Cluster.h"
#include "ClusterList.h"

using namespace std;
using namespace onlineclust;

/////////////////////////// CONSTRUCTOR //////////////////////////////////////////
ClusterList:: ClusterList()
{}

/////////////////////////// DESTRUCTOR /////////////////////////////////////////
ClusterList:: ~ClusterList() 
{}

/////////////////////////// PUBLIC FUNCTIONS ////////////////////////////////////

// Insert a cluster with a given degree and center.
void ClusterList::insertClusterIntoList(unsigned clusterCenterIndex, unsigned clusterSize)
{
  Cluster currCluster(clusterCenterIndex, clusterSize);
  _clusterList.push_back(currCluster);
}

// Return size of the cluster list.
unsigned ClusterList::getClusterListSize()
{
  return _clusterList.size();
}

// Return the list of clusters for printing.
list <Cluster> ClusterList::getSortedClusterListElements()
{
  sortClusterList();
  return _clusterList;
}

// Print to std::cout.
void ClusterList::printClusterList()
{ 
  list <Cluster>::iterator it;
  for (it = _clusterList.begin(); it != _clusterList.end(); ++it)
    {
      Cluster clust = *it;
      clust.printCluster();
    }
}

// Return the cluster center indices as a string. Assumes it has been sorted before.
string ClusterList::stringSortedClusterCenterList()
{
  list <Cluster>::iterator it;
  string s;
    
  // There will be atleast one in the cluster list.
  it = _clusterList.begin();
  Cluster firstCluster = *it;
  s = to_string(firstCluster.getClusterCenterIndex());
    
  // Start from the second.
  it++;
    
  while (it != _clusterList.end())
    {
      Cluster currentCluster = *it;
      s = s + " "+ to_string(currentCluster.getClusterCenterIndex());
      ++it;
    }
    
  return s;
}


// Return the cluster sizes as a string. Assumes it has been sorted before.
string ClusterList::stringSortedClusterSizesList()
{
  list <Cluster>::iterator it;
  string s;
    
  // There will be atleast one in the cluster list.
  it = _clusterList.begin();
  Cluster firstCluster = *it;
  s = to_string(firstCluster.getClusterSize());
    
  // Start from the second.
  it++;
    
  while (it != _clusterList.end())
    {
      Cluster currentCluster = *it;
      s = s + " "+ to_string(currentCluster.getClusterSize());
      ++it;
    }
    
  return s;
}


/////////////////////////// PRIVATE FUNCTIONS ////////////////////////////////////

void ClusterList::sortClusterList()
{
  _clusterList.sort();
}

