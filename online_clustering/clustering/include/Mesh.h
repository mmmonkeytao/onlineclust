#ifndef MESH_H
#define MESH_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace onlineclust{

  class Mesh 
  {
  public:
    
    Mesh();
    
    bool readOFF(char const *filename);
    void computeLP(uint K);
    
    
  private:
    
    std::vector<Eigen::Vector3d>   _verts;
    std::vector<std::vector<uint> > _facets;
    
    std::vector<std::vector<uint> > _v2v;
    std::vector<std::vector<uint> > _v2f;
    
    
    int getVertexIdx(uint face_idx, uint point_id) const;
    Eigen::SparseMatrix<double> computeXuMeyerLaplace(Eigen::VectorXd &d) const;
  };
  
}

#endif
