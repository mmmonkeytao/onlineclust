#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream>

namespace onlineclust {

  class Cluster
  {
        
  public:
    // Constructors
    Cluster();
    Cluster(unsigned clusterCenterIndex, unsigned clusterSize);

        
    // Destructor
    virtual ~Cluster();
        
    //********************        
    // Public functions
    //********************
        
    // Size related.
    unsigned getClusterSize();
    void setClusterSize(unsigned clusterSize);

    // Center index    
    unsigned getClusterCenterIndex();
    void setClusterCenterIndex(unsigned clusterCenterIndex);

    // Print cluster
    void printCluster();
        
        
  private:
    //********************
    //    Private Data
    //********************
    unsigned _clusterSize;
    unsigned _clusterCenterIndex;        
  };

  // Overloaded comparison operator.
  // Note sign is flipped here, so that on sorting the list we get decreasing order.
  inline
  bool operator < (Cluster a, Cluster b)
  {
    return a.getClusterSize() > b.getClusterSize();
  }

}

#endif





