#ifndef CLUSTER_LIST_H
#define CLUSTER_LIST_H

#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <sstream>


#include "Cluster.h"

namespace onlineclust{

  class ClusterList
  {
        
  public:
    // Constructor
    ClusterList();
        
    // Destructor
    virtual ~ClusterList();
        
    //********************
    // Public Functions
    //********************
        
    // Insert a cluster with a given degree and center.
    void insertClusterIntoList(unsigned clusterCenterIndex, unsigned clusterSize);

    // Get size of the cluster list.
    unsigned getClusterListSize();
        
    // To string convertor
    template <class T>
    inline std::string to_string (const T& t)
    {
      std::stringstream ss;
      ss << t;
      return ss.str();
    }
        
    // Return the list of clusters for printing.
    std::list <Cluster> getSortedClusterListElements();
        
    // Print the entire list to std::cout.
    void printClusterList();
        
    // Returns a string for output. Assumes sorted.
    std::string stringSortedClusterCenterList();
    std::string stringSortedClusterSizesList();
        
    // Sort the list by degree.
    void sortClusterList();
        
  private:
        
    //********************
    //    Private Data
    //********************
        
    // Storing clusters in a vector.
    std::list <Cluster> _clusterList;        

  };
}

#endif
