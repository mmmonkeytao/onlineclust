#ifndef ONLINESTARCLUSTERING_H
#define ONLINESTARCLUSTERING_H

#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <math.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <stdio.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Vertex.h"
// #include "Cluster.h"
// #include "ClusterList.h"

using namespace Eigen;

namespace onlineclust {

  typedef SparseVector<double, ColMajor, int> SparseVectorXd;
  typedef std::vector<SparseVectorXd> SparseDataSet;
  
  class OnlineStarClustering  
  {
  public:

    typedef std::vector<Eigen::VectorXd> DataSet;
    
    OnlineStarClustering(double sigma = 0.7);
    OnlineStarClustering(double sigma, uint feaSize);
    
    // Destructor
    virtual ~OnlineStarClustering();    
    void addPoint(Eigen::VectorXd const &p, uint idx);
    void clear();

    void exportDot(char const *filename, bool use_data) const;

    // new codes
    /////////////////////////////////////////////////////////
    void loadAndAddData(char const *filename);
    void deleteData(std::list<uint> deleteList);
    void V_measure(double beta);
    uint getDataSize();
    /* add for sparse case */
    void insert(VectorXd &sv);
    void insertSparseData();
    //////////////////////////////////////////////////////////
  protected:

    virtual double computeSimilarity(Eigen::VectorXd const &x1,
		                     Eigen::VectorXd const &x2) const;
    virtual double computeSimilarity(SparseVectorXd const &x1, 
				     SparseVectorXd const &x2 )const; 

  private:

    typedef Eigen::MatrixXd MatrixType;

    DataSet _data;
    MatrixType _similarityMatrix;

    double _sigma;  
    
    std::map <uint, Vertex> _graph; // Thresholded graph.
    std::map<uint, uint>  _id2idx;  // maps from vertex ID to index
    std::list <uint> _elemInGraphSigma;
    unsigned _numClusters;
    ///////////////////////////////////////////////
    std::map<uint, double> _labels;
    /* add for sparse case */
    SparseDataSet _spData;
    uint _feaSize;  // store the feature size
    ///////////////////////////////////////////////

    std::priority_queue<Vertex> _priorityQ;

    void insertData(uint start_idx);
    void insertLastPoint(uint new_vertex_id);

    void fastInsert(uint alphaID, std::list <uint> &L);
    void fastUpdate(uint alphaID);
    Eigen::VectorXd readVector(std::ifstream &ifile) const;
    void updateSimilarityMatrix(uint start_idx);
    uint vertexIDMaxDeg(std::list<uint> const &L) const;

    void fastDelete(uint alphaID);
    void sortList(std::list <uint>& AdjCV);

    /*
     *  new added codes for Eigen::SparseVector data insertion, need to be integrated  
     */
    

  };
  
}


#endif
