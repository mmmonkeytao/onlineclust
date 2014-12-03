#include "Mesh.h"
#include <iostream>
#include <fstream>

using namespace onlineclust;
using namespace std;

Mesh::Mesh() : _verts(), _facets()
{}

bool Mesh::readOFF(char const *filename)
{
  ifstream ifile(filename);
  string tag;
  ifile >> tag;
  if(tag != "OFF")
    return false;

  uint nb_verts, nb_facets, dummy, len;
  ifile >> nb_verts >> nb_facets >> dummy;

  _verts.resize(nb_verts);
  _facets.resize(nb_facets);
  
  for(uint i=0; i<nb_verts; ++i)
    ifile >> _verts[i][0] >> _verts[i][1] >> _verts[i][2];

  for(uint i=0; i<nb_facets; ++i){
    ifile >> len;
    _facets[i].resize(len);
    for(uint j=0; j<len; ++j)
      ifile >> _facets[i][j];
  }

  _v2v.resize(nb_verts);
  _v2f.resize(nb_verts);

  for(uint i=0; i<nb_facets; ++i){
    for(uint j=0; j<_facets[i].size(); ++j){
      uint v1 = _facets[i][j];
      uint v2 = _facets[i][(j+1) % _facets[i].size()];
      
      _v2v[v1].push_back(v2);
      _v2f[v1].push_back(i);
    }
  }

  return true;
}

void Mesh::computeLP(uint K)
{
  Eigen::VectorXd d;

  Eigen::MatrixXd W = -computeXuMeyerLaplace(d);
  std::cout << W << std::endl;

  uint n = d.size();

  Eigen::MatrixXd A(n,n);
  A.setZero();
  for(uint i=0; i<n; ++i)
    A(i,i) = d(i);
	
  std::cout << W << std::endl;
  std::cout << A << std::endl;

  //uint nb_evals = static_cast<uint>(d.size()) > K ? K : d.size();

  typedef Eigen::MatrixXd MatrixType;
  Eigen::GeneralizedEigenSolver<MatrixType> solver;
  solver.compute(W, A);

  std::cout << solver.eigenvalues() << std::endl;
}

int Mesh::getVertexIdx(uint face_idx, uint point_id) const
{
  for(uint i=0; i<_facets[face_idx].size(); ++i)
    if(_facets[face_idx][i] == point_id)
      return (int)i;

  return -1;
}

Eigen::SparseMatrix<double> Mesh::computeXuMeyerLaplace(Eigen::VectorXd &ddv) const
{
  uint nv = _verts.size();
  uint nf = _facets.size();

 //compute the center and area for each facets
  vector<double> fareas(3*nf), fcots(3*nf);

  uint vid0, vid1, vid2;

  for(uint f = 0; f < nf; f ++){
    vid0 = _facets[f][0];
    vid1 = _facets[f][1];
    vid2 = _facets[f][2];

    Eigen::Vector3d vv = (_verts[vid1] - _verts[vid0]).cross(_verts[vid2] - _verts[vid0]);
				
    double area = vv.norm();
    double l0 = sqrt( fabs((_verts[vid1] - _verts[vid2]).dot(_verts[vid1] - _verts[vid2])) );
    double l1 = sqrt( fabs((_verts[vid0] - _verts[vid2]).dot(_verts[vid0] - _verts[vid2])) );
    double l2 = sqrt( fabs((_verts[vid1] - _verts[vid0]).dot(_verts[vid1] - _verts[vid0])) );

    //assert(l0 > MYNZERO && l1 > MYNZERO && l2 > MYNZERO);

    double h0 = area / l0;
    double h1 = area / l1;
    double h2 = area / l2;

    fcots[3 * f] = sqrt(fabs(l1 * l1 - h2 * h2)) / h2;
    if( (l1 * l1 + l2 * l2 - l0 * l0) < 0){
      fcots[3 * f] = -fcots[3 * f];
    }
    fcots[3 * f + 1] = sqrt(fabs(l2 * l2 - h0 * h0)) / h0;
    if( (l0 * l0 + l2 * l2 - l1 * l1) < 0){
      fcots[3 * f + 1] = -fcots[3 * f + 1];
    }
    fcots[3 * f + 2] = sqrt(fabs(l0 * l0 - h1 * h1)) / h1;
    if( (l0 * l0 + l1 * l1 - l2 * l2) < 0 ){
      fcots[3 * f + 2] = -fcots[3 * f + 2];
    }

    if( fcots[3 * f] >= 0 && fcots[3 * f + 1] >= 0 && fcots[3 * f + 2] >= 0 ){
      fareas[3 * f] = 1.0 / 8.0 * (l1 * l1 * fcots[3 * f + 1] + l2 * l2 * fcots[3 * f + 2]);
      fareas[3 * f + 1] = 1.0 / 8.0 * (l0 * l0 * fcots[3 * f] + l2 * l2 * fcots[3 * f + 2]);
      fareas[3 * f + 2] = 1.0 / 8.0 * (l1 * l1 * fcots[3 * f + 1] + l0 * l0 * fcots[3 * f]);
    }
    else if(fcots[3 * f] < 0){
      fareas[3 * f] = area / 4;
      fareas[3 * f + 1] = area / 8;
      fareas[3 * f + 2] = area / 8;
    }
    else if(fcots[3 * f + 1] < 0){
      fareas[3 * f] = area / 8;
      fareas[3 * f + 1] = area / 4;
      fareas[3 * f + 2] = area / 8;
    }
    else{
      fareas[3 * f] = area / 8;
      fareas[3 * f + 1] = area / 8;
      fareas[3 * f + 2] = area / 4;
    }
  }

  vector<Eigen::Triplet<double> > tripletList;
  double Amix, totalweight;
  map<uint, double> adj_weight;

  ddv = Eigen::VectorXd(nv);

  for(uint i = 0; i < nv; ++i){
    Amix = 0;
    totalweight = 0;
    adj_weight.clear();

    for(uint kk = 0; kk < _v2v[i].size(); ++kk){
      adj_weight[_v2v[i][kk]] = 0;
    }

    for(uint kk = 0; kk < _v2f[i].size(); ++kk){
      uint f = _v2f[i][kk];
      int ind = getVertexIdx(f, i);
      
      Amix += fareas[3 * f + ind];
      
      vid1 = _facets[f][(ind + 1) % 3];
      vid2 = _facets[f][(ind + 2) % 3];

      std::cout << "at1" << std::endl;
      adj_weight.at(vid1) += fcots[3 * f + (ind + 2) % 3];
      std::cout << "at2" << std::endl;
      adj_weight.at(vid2) += fcots[3 * f + (ind + 1) % 3];
      std::cout << "at3" << std::endl;

      totalweight -= (fcots[3 * f + (ind + 2) % 3] + fcots[3 * f + (ind + 1) % 3]);
    }

    Amix *= 2;
    
    for(map<uint, double>::const_iterator iter = adj_weight.begin(); 
	iter != adj_weight.end(); ++iter){

      tripletList.push_back(Eigen::Triplet<double>(i, iter->first, iter->second));
    }

    tripletList.push_back(Eigen::Triplet<double>(i, i, totalweight));    
    ddv(i) = Amix;
  }	

  Eigen::SparseMatrix<double> mat(nv, nv);
  mat.setFromTriplets(tripletList.begin(), tripletList.end());

  return mat;
}


