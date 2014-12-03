#include "GPlib/GP_Tensor.hh"
#include "GPlib/GP_Exception.hh"
#include <sstream>

namespace GPLIB {

  uint GP_Tensor::Rows() const
  {
    return _data.size();
  }

  uint GP_Tensor::Cols() const
  {
    if(_data.size() == 0)
      return 0;
  
    return _data[0].size();
  }

  uint GP_Tensor::Layers() const
  {
    if(_data.size() == 0)
      return 0;

    if(_data[0].size() == 0)
      return 0;
  
    return _data[0][0].size();
  }

  GP_Matrix GP_Tensor::Row(uint idx) const
  {
    if(idx >= _data.size()){
      std::stringstream msg;
      msg << idx << " != " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }

    uint n = Cols(), l = Layers();
    GP_Matrix row_mat(n, l);

    for(uint j=0; j<n; ++j)
      for(uint k=0; k<l; ++k)
	row_mat[j][k] = _data[idx][j][k];

    return row_mat;
  }

  GP_Matrix GP_Tensor::Col(uint idx) const
  {
    if(idx >= Cols()){
      std::stringstream msg;
      msg << idx << " != " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }

    uint m = Rows(), l = Layers();
    GP_Matrix col_mat(m, l);

    for(uint i=0; i<m; ++i)
      for(uint k=0; k<l; ++k)
	col_mat[i][k] = _data[i][idx][k];

    return col_mat;
  }

  GP_Matrix GP_Tensor::Layer(uint idx) const
  {
    if(idx >= Layers()){
      std::stringstream msg;
      msg << idx << " != " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }

    uint m = Rows(), n = Cols();
    GP_Matrix lay_mat(m, n);

    for(uint i=0; i<m; ++i)
      for(uint j=0; j<n; ++j)
	lay_mat[i][j] = _data[i][j][idx];

    return lay_mat;
  }

  GP_Vector GP_Tensor::operator()(uint i, uint j, all) const
  {
    if(i >= Rows() || j >= Cols())
      throw GP_EXCEPTION("Invalid index into tensor");

    uint l = Layers();
    GP_Vector out(l);

    for(uint k=0; k<l; ++k)
      out[k] = _data[i][j][k];

    return out;
  }

  GP_Vector GP_Tensor::operator()(uint i, all, uint k) const
  {
    if(i >= Rows() || k >= Layers())
      throw GP_EXCEPTION("Invalid index into tensor");

    uint n = Cols();
    GP_Vector out(n);

    for(uint j=0; j<n; ++j)
      out[j] = _data[i][j][k];

    return out;
  }

  GP_Vector GP_Tensor::operator()(all, uint j, uint k) const
  {
    if(j >= Cols() || k >= Layers())
      throw GP_EXCEPTION("Invalid index into tensor");

    uint m = Rows();
    GP_Vector out(m);

    for(uint i=0; i<m; ++i)
      out[i] = _data[i][j][k];

    return out;
  }


  std::vector<std::vector<double> > const &
  GP_Tensor::operator[](uint i) const
  {
    if(i >= _data.size()){
      std::stringstream msg;
      msg << i << " >= " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }
    return _data[i];
  }

  std::vector<std::vector<double> > &GP_Tensor::operator[](uint i)
  {
    if(i >= _data.size()){
      std::stringstream msg;
      msg << i << " >= " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }

    return _data[i];
  }

}
