#include "GPlib/GP_Matrix.hh"
#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_Exception.hh"

#include <math.h>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_eigen.h>

#include <string.h>

namespace GPLIB {

  uint GP_Matrix::Rows() const
  {
    return _data.size();
  }

  uint GP_Matrix::Cols() const
  {
    if(_data.size() == 0)
      return 0;

    return _data[0].size();
  }

  GP_Vector GP_Matrix::Row(uint idx) const
  {
    if(idx >= _data.size()){
      std::stringstream msg;
      msg << idx << " != " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }

    GP_Vector row_vec(_data[idx].size());
    for(uint i=0; i<row_vec.Size(); ++i)
      row_vec[i] = _data[idx][i];

    return row_vec;
  }

  GP_Vector GP_Matrix::Col(uint idx) const
  {
    if(idx >= Cols()){
      std::stringstream msg;
      msg << idx << " != " << Cols();
      throw GP_EXCEPTION2("Invalid column index %d into matrix", msg.str());
    }

    GP_Vector col_vec(_data.size());
    for(uint i=0; i<col_vec.Size(); ++i)
      col_vec[i] = _data[i][idx];

    return col_vec;
  }


  GP_Matrix GP_Matrix::Transp() const
  {
    GP_Matrix out(Cols(), Rows());
    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	out[j][i] = _data[i][j];

    return out;
  }

  GP_Matrix GP_Matrix::SubMatrix(uint x1, uint y1, uint x2, uint y2) const
  {
    if(x2 > Cols() || y2 > Rows() ||
       x1 > x2 || y1 > y2)
      throw GP_EXCEPTION("Invalid indices of sub matrix");

    GP_Matrix sub(y2-y1, x2-x1);
    for(uint i=y1; i < y2; ++i)
      for(uint j=x1; j < x2; ++j){
	sub[i-y1][j-x1] = _data[i][j];
      }

    return sub;
  }

  std::vector<double> const &GP_Matrix::operator[](uint i) const
  {
    if(i >= _data.size()){
      std::stringstream msg;
      msg << i << " >= " << _data.size();
      throw GP_EXCEPTION2("Invalid row index %d into matrix", msg.str());
    }
    return _data[i];
  }

  std::vector<double> &GP_Matrix::operator[](uint i)
  {
    if(i >= _data.size()){
      std::stringstream msg;
      msg << i << " != " << _data.size();
      throw GP_EXCEPTION2("Invalid row index into matrix: %s", msg.str());
    }
    return _data[i];
  }

  GP_Matrix GP_Matrix::operator+(GP_Matrix const &other) const
  {
    if(other.Rows() != Rows() || other.Cols() != Cols())
      throw GP_EXCEPTION("Matrix dimensions do not match.");
    
    GP_Matrix out(Rows(), Cols());

    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	out[i][j] = _data[i][j] + other[i][j];

    return out;
  }
  
  GP_Matrix GP_Matrix::operator-(GP_Matrix const &other) const
  {
    if(other.Rows() != Rows() || other.Cols() != Cols())
      throw GP_EXCEPTION("Matrix dimensions do not match.");

    GP_Matrix out(Rows(), Cols());

    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	out[i][j] = _data[i][j] - other[i][j];

    return out;
  }
  
  GP_Matrix &GP_Matrix::operator+=(GP_Matrix const &other)
  {
    if(other.Rows() != Rows() || other.Cols() != Cols())
      throw GP_EXCEPTION("Matrix dimensions do not match.");
    
    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	_data[i][j] += other[i][j];

    return *this;
  }
  
  GP_Matrix &GP_Matrix::operator-=(GP_Matrix const &other)
  {
    if(other.Rows() != Rows() || other.Cols() != Cols())
      throw GP_EXCEPTION("Matrix dimensions do not match.");
    
    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	_data[i][j] -= other[i][j];
    
    return *this;
  }

  GP_Matrix GP_Matrix::ElemMult(GP_Matrix const &other) const
  {
    if(other.Rows() != Rows() || other.Cols() != Cols())
      throw GP_EXCEPTION("Matrix dimensions do not match.");

    GP_Matrix out(Rows(), Cols());

    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	out[i][j] = _data[i][j] * other[i][j];

    return out;
  }
  
  GP_Matrix GP_Matrix::ElemDiv(GP_Matrix const &other) const
  {
    if(other.Rows() != Rows() || other.Cols() != Cols())
      throw GP_EXCEPTION("Matrix dimensions do not match.");

    GP_Matrix out(Rows(), Cols());

    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	out[i][j] = _data[i][j] / other[i][j];

    return out;
  }


  GP_Vector GP_Matrix::operator*(GP_Vector const &v) const
  {
    if(v.Size() != Cols()){
      std::stringstream msg;
      msg << v.Size() << " != " << Cols();
      throw GP_EXCEPTION2("Vector length does not match number of columns",
			  msg.str());
    }

    GP_Vector out(_data.size());
    for (uint row = 0; row < _data.size(); row++){
      out[row] = 0;
      for (uint col = 0; col < v.Size(); col++)
	out[row] += _data[row][col] * v[col];
    }

    return out;
  }
  

  GP_Matrix GP_Matrix::operator* (double x) const
  {
    GP_Matrix prod;
    prod._data = _data;

    for (uint row = 0; row < Rows(); row++)
      for (uint col = 0; col < Cols(); col++)
	prod[row][col] = x * _data[row][col];
  
    return prod;
  }

  GP_Matrix GP_Matrix::operator*(GP_Matrix const &other) const
  {
    if (Cols() != other.Rows() )
      throw GP_EXCEPTION("Matrix dimensions do not agree.");

    GP_Matrix prod(Rows(), other.Cols());
    for (uint row = 0; row < Rows(); row++)
      for (uint col = 0; col < other.Cols(); col++){
	prod[row][col] = 0.;
	for (uint i=0; i < other.Rows(); i++)
	  prod[row][col] += _data[row][i] * other[i][col];
      }

    return prod;
  }

  GP_Matrix GP_Matrix::operator/ (double x) const
  {
    if(fabs(x) < 1e-15)
      throw GP_EXCEPTION("Division by zero.");

    GP_Matrix prod;
    prod._data = _data;

    for (uint row = 0; row < Rows(); row++)
      for (uint col = 0; col < Cols(); col++)
	prod[row][col] = _data[row][col] / x;
  
    return prod;
  }


  GP_Matrix GP_Matrix::TranspTimes(GP_Matrix const &other)  const
  {
    uint rows_in = Rows(), rows_out = Cols(), cols_out = other.Cols();
    
    if (rows_in == 0)
      return GP_Matrix();
    
    if (rows_in != other.Rows() )
      throw GP_EXCEPTION("Matrix dimensions do not agree.");
    
    GP_Matrix prod(rows_out, cols_out);
    for (uint row = 0; row < rows_out; row++)
      for (uint col = 0; col < cols_out; col++){
	prod._data[row][col] = 0.;
	for (uint i=0; i < rows_in; i++)
	  prod._data[row][col] += _data[i][row] * other._data[i][col];
      }
    
    return prod;
  }

  GP_Matrix GP_Matrix::TimesTransp(GP_Matrix const &other)  const
  {
    if (Rows() == 0)
      return GP_Matrix();
    
    if (Cols() != other.Cols() )
      throw GP_EXCEPTION("Matrix dimensions do not agree.");
    
    GP_Matrix prod(Rows(), other.Rows());
    for (uint row = 0; row < prod.Rows(); row++)
      for (uint col = 0; col < prod.Cols(); col++){
	prod[row][col] = 0.;
	for (uint j=0; j < Cols(); j++)
	  prod[row][col] += _data[row][j] * other[col][j];
      }
    
    return prod;
  }


  double GP_Matrix::Fnorm() const
  {
    double sumsq = 0.0;

    for (uint i = 0; i < _data.size(); i++ )
      for (uint j = 0; j < _data[i].size(); j++ )
	sumsq += _data[i][j] * _data[i][j];

    return sqrt(sumsq);
  }

  GP_Vector GP_Matrix::Diag() const
  {
    GP_Vector diag(MIN(Rows(), Cols()));

    for(uint i=0; i<diag.Size(); i++)
      diag[i] = _data[i][i];

    return diag;
  }

  GP_Matrix GP_Matrix::Diag(GP_Vector const &elems) 
  {
    GP_Matrix diag(elems.Size(), elems.Size());

    for(uint i=0; i<elems.Size(); i++)
      diag[i][i] = elems[i];

    return diag;
  }

  gsl_matrix *GP_Matrix::MakeGSLMatrix() const
  {
    uint m = Rows();
    uint n = Cols();

    gsl_matrix *mat = gsl_matrix_alloc(m, n);
    for(uint i=0; i<m; ++i)
      for(uint j=0; j<n; ++j)
	gsl_matrix_set(mat, i, j, _data[i][j]);

    return mat;
  }

  GP_Matrix const &GP_Matrix::Invert()
  {
    if (Rows() != Cols() ) 
      throw GP_EXCEPTION("Matrix must be square.");

    int s;
    uint m = Rows();
    uint n = Cols();

    gsl_matrix *mat = MakeGSLMatrix();
    gsl_matrix *inv = gsl_matrix_alloc(m, n);
    gsl_permutation * p = gsl_permutation_alloc (m);

    for(uint i=0; i<m; ++i)
      for(uint j=0; j<n; ++j)
	gsl_matrix_set(mat, i, j, _data[i][j]);

    gsl_linalg_LU_decomp(mat, p, &s);
    gsl_linalg_LU_invert(mat, p, inv);

    for(uint i=0; i<m; ++i)
      for(uint j=0; j<n; ++j)
	_data[i][j]  = gsl_matrix_get(inv, i, j);

    gsl_matrix_free(mat);
    gsl_matrix_free(inv);
    gsl_permutation_free(p);

    return *this;    
  }

  GP_Matrix GP_Matrix::Inverse() const
  {
    GP_Matrix Inv(*this);
    Inv.Invert();

    return Inv;
  }

  GP_Matrix GP_Matrix::InverseByChol() const
  {
    GP_Matrix out = *this;

    out.Cholesky();
    out = out.ForwSubst(Identity(Rows()));

    return out.TranspTimes(out);
  }

  GP_Matrix GP_Matrix::Identity(uint m)
  {
    return Diag(GP_Vector(m,1.));
  }


  double GP_Matrix::Sum() const
  {
    double sum = 0;
    
    for(uint i=0; i<_data.size(); i++)
      for(uint j=0; j<_data[i].size(); j++)
	sum += _data[i][j];

    return sum;
  }

  double GP_Matrix::Trace() const
  {
    double tr = 0.0;
    
    for(uint i=0; i<MIN(Rows(), Cols()); i++)
      tr += _data[i][i];
    
    return tr;
  }

  double GP_Matrix::LogTrace() const
  {
    double tr = 0.0;
    
    for(uint i=0; i<MIN(Rows(), Cols()); i++){
      if(_data[i][i] < 0)
	throw GP_EXCEPTION("Log Trace only works for positive diagonal");
      tr += log(_data[i][i]);
    }
    
    return tr;
  }

  double GP_Matrix::Max() const
  {
    double max = -HUGE_VAL;

    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	max = MAX(max, _data[i][j]);
    
    return max;
  }
  
  double GP_Matrix::Min() const
  {
    double min = HUGE_VAL;

    for(uint i=0; i<_data.size(); ++i)
      for(uint j=0; j<_data[i].size(); ++j)
	min = MIN(min, _data[i][j]);
    
    return min;
  }



  GP_Matrix GP_Matrix::OutProd(GP_Vector const &v)
  {
    GP_Matrix out(v.Size(), v.Size());
    
    for(uint i=0; i<v.Size(); i++)
      for(uint j=0; j<v.Size(); j++)
	out[i][j] = v[i] * v[j];
    
    return out;
  }

  void GP_Matrix::SVD(GP_Matrix &U, GP_Matrix &D, GP_Matrix &VT) const
  {
    if (Rows() == 0 || Cols() == 0) return;
    if (Rows() < Cols())
      throw GP_EXCEPTION("SVD of MxN matrix with m < n not implemented.");

    uint m = Rows();
    uint n = Cols();

    gsl_matrix *u  = MakeGSLMatrix();
    gsl_vector *dd = gsl_vector_alloc(n);
    gsl_vector *wk = gsl_vector_alloc(n);
    gsl_matrix *v  = gsl_matrix_alloc(n, n);

    gsl_linalg_SV_decomp(u, v, dd, wk);
    gsl_matrix_transpose(v);

    U  = GP_Matrix(m, n);
    D  = GP_Matrix(n, n);
    VT = GP_Matrix(n, n);

    for(uint i=0; i<m; ++i)
      for(uint j=0; j<n; ++j)
	U._data[i][j] = gsl_matrix_get(u, i, j);

    for(uint i=0; i<n; i++)
      D[i][i] = gsl_vector_get(dd, i);

    for(uint i=0; i<n; ++i)
      for(uint j=0; j<n; ++j)
	VT._data[i][j] = gsl_matrix_get(v, i, j);

    gsl_matrix_free(u);
    gsl_vector_free(dd);
    gsl_vector_free(wk);
    gsl_matrix_free(v);
  }

  
  bool GP_Matrix::Cholesky()
  {
    uint rows = _data.size();
    std::vector<double> diag(rows);
    double sum;

    for(uint i=0; i<rows; i++)
      for(uint j=i; j<rows; j++){
	sum = _data[i][j];

	for(int k=(int)i-1; k>= 0; k--)
	  sum -= _data[i][k] * _data[j][k];
	  
	if(i==j){
	  if(sum <= 0.0)
	    return false;
	  diag[i] = sqrt(sum);
	}
	else
	  _data[j][i] = sum / diag[i];
      }
      
    for(uint i=0; i<rows; i++)
      for(uint j=i; j<rows; j++)
	if(i==j)
	  _data[i][j] = diag[i];
	else
	  _data[i][j] = 0;

    return true;
  }

  double GP_Matrix::SumLogDiag() const
  {
    double sum = 0;
    uint m = Rows(), n = Cols();
    for(uint i=0; i<MIN(m, n); ++i)
      sum += log(_data[i][i]);

    return sum;
  }

  double GP_Matrix::Det() const
  {
    if (Rows() != Cols()) return 0;
    uint n = Rows();

    int s;
    double dd;
    gsl_matrix *a = MakeGSLMatrix();
    gsl_permutation *p = gsl_permutation_alloc(n);

    gsl_linalg_LU_decomp(a, p, &s);
    dd = gsl_linalg_LU_det(a, s);

    gsl_matrix_free(a);
    gsl_permutation_free(p);

    return dd;
  }

  GP_Matrix GP_Matrix::Abs() const
  {
    GP_Matrix out(*this);

    for(uint i=0; i<out._data.size(); ++i)
      for(uint j=0; j<out._data[i].size(); ++j)
	out._data[i][j] = fabs(out._data[i][j]);

    return out;
  }


  // Compute Cholesky decomp of Matrix B = I + W^0.5 * K * W^0.5 
  // W is a diagonal matrix and is represented only as a vector
  void GP_Matrix::CholeskyDecompB(GP_Vector const &w_sqrt)
  {
    if(w_sqrt.Size() != Rows() || w_sqrt.Size() != Cols())
      throw GP_EXCEPTION("Vector size must match number of cols and rows");

    uint n = w_sqrt.Size();
    
    for(uint i=0; i<n; i++)
      for(uint j=0; j<n; j++){
	_data[i][j] *= w_sqrt[i] * w_sqrt[j];
	if(i == j)
	  _data[i][j] += 1.0;
      }
    
    if(!Cholesky()){
      throw GP_EXCEPTION("Could not perform Cholesky decomp: "
			 "Matrix is not positive definite.");
    }
  }

  GP_Vector GP_Matrix::ForwSubst(GP_Vector const &b) const
  {
    GP_Vector x(Rows());

    for(uint i=0; i<x.Size(); ++i){
      double sum = b[i];
      for(int k = (int)i-1; k >= 0; --k)
	sum -= _data[i][k] * x[k];
      x[i] = sum / _data[i][i];
    }
    return x;
  }

  GP_Vector GP_Matrix::BackwSubst(GP_Vector const &b) const
  {
    GP_Vector x(Rows());

    for(int i=(int)x.Size() - 1; i >= 0; --i){
      double sum = b[i];
      for(int k = i+1; k < (int)x.Size(); ++k)
	sum -= _data[k][i] * x[k];
      x[i] = sum / _data[i][i];
    }

    return x;
  }

  GP_Vector GP_Matrix::SolveChol(GP_Vector const &b) const
  {
    return BackwSubst(ForwSubst(b));
  }

  GP_Matrix GP_Matrix::ForwSubst(GP_Matrix const &B) const
  {
    uint m = Rows(), n = B.Cols();
    GP_Matrix X(m, n);

    for(uint j=0; j<n; j++)
      for(uint i=0; i<m; i++){
	double sum = B._data[i][j];
	for(int k=(int)i-1; k>= 0; k--)
	  sum -= _data[i][k] * X._data[k][j];
	X._data[i][j] = sum / _data[i][i];
      }
    return X;
  }

  GP_Matrix GP_Matrix::TForwSubst(GP_Matrix const &B) const
  {
    GP_Matrix X(Cols(), B.Cols());

    for(uint j=0; j<B.Cols(); j++)
      for(uint i=0; i<Cols(); i++){
	double sum = B[i][j];
	for(int k=(int)i-1; k>= 0; k--)
	  sum -= _data[k][i] * X[k][j];
	X[i][j] = sum / _data[i][i];
      }
    return X;
  }

  GP_Matrix GP_Matrix::BackwSubst(GP_Matrix const &B) const
  {
    GP_Matrix X(Rows(), B.Cols());

    for(uint j=0; j<B.Cols(); j++)
      for(int i=(int)Rows()-1; i>=0; i--){
	double sum = B[i][j];
	for(int k=i+1; k<(int)Rows(); k++)
	  sum -= _data[k][i] * X[k][j];
	X[i][j] = sum / _data[i][i];
      }

    return X;
  }

  GP_Matrix GP_Matrix::SolveChol(GP_Matrix const &B) const
  {
    return BackwSubst(ForwSubst(B));
  }


  void GP_Matrix::EigenSolveSymm(GP_Vector &eval, GP_Matrix &evec, bool sortflag) const
				 
  {
    if(Rows() != Cols())
      throw GP_EXCEPTION("Eigen value decomposition of non-square matrix");

    uint m = Rows();

    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc (m);
    gsl_matrix *a   = gsl_matrix_alloc(m, m);
    gsl_vector *val = gsl_vector_alloc (m);
    gsl_matrix *vec = gsl_matrix_alloc (m, m);

    double *entries = new double[m*m];
    uint k=0;
    for(uint i=0; i<m; i++)
      for(uint j=0; j<m; j++)
	if (j>= i)
	  entries[i*m+j] = _data[i][j];
	else
	  entries[i*m+j] = entries[j*m+i];
    
    k=0;
    memcpy(a->data, entries, m * m * sizeof(double));
    gsl_eigen_symmv (a, val, vec, w);
    
    delete[] entries;
    
    if (sortflag)
      gsl_eigen_symmv_sort (val, vec, GSL_EIGEN_SORT_VAL_DESC);
    else
      gsl_eigen_symmv_sort (val, vec, GSL_EIGEN_SORT_VAL_ASC);
    
    eval = GP_Vector(m);
    evec = GP_Matrix(m, m);

    for(uint i=0; i<m; ++i){
      eval[i] = val->data[i];
      for(uint j=0; j<m; ++j){
	evec._data[i][j] = vec->data[i*m+j];
      }
    }

    gsl_eigen_symmv_free(w);
    gsl_matrix_free(a);
    gsl_vector_free(val);
    gsl_matrix_free(vec);
  }
  


  GP_Vector GP_Matrix::RowSum() const
  {
    if(Rows() == 0)
      return GP_Vector();

    GP_Vector sum = Row(0);
    for(uint i=1; i<Rows(); ++i)
      sum += Row(i);

    return sum;
  }

  void GP_Matrix::AppendRow(GP_Vector const &row)
  {
    if (Cols() != row.Size()) 
      throw GP_EXCEPTION("Matrix/Vector dimensions do not agree.");

    _data.push_back(std::vector<double>(row.Size()));
    for(uint i=0; i<row.Size(); ++i)
      _data.back()[i] = row[i];
  }
  
  void GP_Matrix::AppendColumn(GP_Vector const &col)
  {
    if (Rows() != col.Size()) 
      throw GP_EXCEPTION("Matrix/Vector dimensions do not agree.");

    for(uint i=0; i<_data.size(); ++i)
      _data[i].push_back(col[i]);
  }

  void GP_Matrix::AppendVert(GP_Matrix const &other)
  {
    if(_data.size() != other._data.size())
      throw GP_EXCEPTION("Matrices must have same number of rows.");

    for(uint i=0; i<_data.size(); ++i){
      uint old_cols = _data[i].size();
      uint new_cols = old_cols + other._data[i].size();
      _data[i].resize(new_cols);
      for(uint j=0; j<other._data[i].size(); ++j)
	_data[i][j+old_cols] = other._data[i][j];
    }
  }

  void GP_Matrix::AppendHor(GP_Matrix const &other)
  {
    if(Cols() != other.Cols())
      throw GP_EXCEPTION("Matrices must have same number of columns.");

    for(uint i=0; i<other._data.size(); ++i){
      _data.push_back(other._data[i]);
    }
  }

  void GP_Matrix::RemoveLastRow()
  {
    _data.pop_back();
  }

  void GP_Matrix::RemoveLastColumn()
  {
    for(uint i=0; i<_data.size(); ++i)
      _data[i].pop_back();
  }

  GP_Matrix GP_Matrix::RemoveRowAndColumn(uint idx) const
  {
    uint m = Rows(), n= Cols();
    if(idx >= m || idx >= n)
      throw GP_EXCEPTION("Invalid row/column index");

    GP_Matrix out(m-1, n-1);
    uint row_shift = 0, col_shift = 0;
    for(uint i=0; i<m-1; ++i)
      for(uint j=0; j<n-1; ++j){

	if(i < idx)
	  row_shift = 0;
	else
	  row_shift = 1;
	if(j < idx)
	  col_shift = 0;
	else
	  col_shift = 1;

	out[i][j] = _data[i+row_shift][j+col_shift];
      }
	  
    return out;
  }

  int GP_Matrix::Read(std::string filename, int pos)
  {
    READ_FILE(ifile, filename.c_str());
    ifile.seekg(pos);

    uint rows, cols;
    ifile >> rows >> cols;
    _data.resize(rows, std::vector<double>(cols));
    for(uint i=0; i<rows; ++i){
      for(uint j=0; j<cols; ++j){
	ifile >> _data[i][j];
	std::cout << _data[i][j] << " " << std::flush;
      }
      std::cout << std::endl;
    }

    return ifile.tellg();
  }

  void GP_Matrix::Write(std::string filename) const
  {
    APPEND_FILE(ofile, filename.c_str());
    ofile << _data.size() << " " 
	  << (_data.size() == 0 ? 0 : _data[0].size()) << std::endl;
    for(uint i=0; i<_data.size(); ++i){
      for(uint j=0; j<_data[i].size(); ++j)
	ofile << std::fixed << std::setw(5) << _data[i][j] << " ";
      ofile << std::endl;
    }
    ofile << std::endl;
    ofile.close();
  }


GP_Matrix GP_Matrix::Kronecker(GP_Matrix const &other, uint m, uint n)
{
  uint rows = other.Rows();
  uint cols = other.Cols();
  GP_Matrix out(rows * m, cols * n);

  for(uint i=0; i<m; ++i)
    for(uint j=0; j<n; ++j)
      for(uint k=0; k<rows; ++k)
	for(uint l=0; l<cols; ++l)
	  out[i*rows+k][j*cols+l] = other[k][l];

  return out;
}

GP_Matrix GP_Matrix::Kronecker(GP_Matrix const &other) const
{
  uint orows = other.Rows();
  uint ocols = other.Cols();
  uint trows = Rows();
  uint tcols = Cols();
  
  GP_Matrix out(orows * trows, ocols * tcols);

  for(uint i=0; i<trows; ++i)
    for(uint j=0; j<tcols; ++j)
      for(uint k=0; k<orows; ++k)
	for(uint l=0; l<ocols; ++l)
	  out[i*orows+k][j*ocols+l] = other[k][l] * (*this)[i][j];

  return out;
}


  void GP_Matrix::ExportPGM(std::string filename) const
  {
    DEL_FEXT(filename);

    WRITE_FILE(ofile, (filename + ".pgm").c_str());
    ofile << "P2" << std::endl;
    ofile << _data.size() << " " 
	  << (_data.size() == 0 ? 0 : _data[0].size()) << std::endl;
    ofile << "255" << std::endl;
    
    double max = Max();
    double min = Min();
    double range = max - min;
    
    for(uint i=0; i<_data.size(); ++i){
      for(uint j=0; j<_data[i].size(); ++j){
	unsigned int val = (unsigned int) floor(255 * (_data[i][j] - min) / range + 0.5);
	ofile << val << " ";
      }
      ofile << std::endl;
    }
  }

  GP_Matrix operator* (double x, GP_Matrix const &m)
  {
    return m * x;
  }

  GP_Matrix operator/ (double x, GP_Matrix const &m)
  {
    GP_Matrix out(m.Rows(), m.Cols());

    for(uint i=0; i<out.Rows(); ++i)
      for(uint j=0; j<out.Cols(); ++j){

	if(fabs(m[i][j]) < 1e-15)
	  throw GP_EXCEPTION("Division by zero.");

	out[i][j] = x / m[i][j];
      }

    return out;
  }

  GP_Vector operator* (GP_Vector const &vec, GP_Matrix const &mat)
  {
    if (vec.Size() != mat.Rows())
      throw GP_EXCEPTION("Matrix/Vector dimensions do not agree.");
    
    GP_Vector prod(mat.Cols());
    for (uint col = 0; col < mat.Cols(); col++){
      prod[col] = 0.;
      for (uint row = 0; row < mat.Rows(); row++)
	prod[col] += vec[row] * mat[row][col];
    }
  
    return prod;
  }

  std::ostream &operator<<(std::ostream &stream, GP_Matrix const &mat)
  {
    uint prec = stream.precision();
    prec = stream.precision(MAX(3, prec));
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream << mat.Rows() << "x" <<mat.Cols() << " Matrix:" << std::endl;
    for(uint i=0; i<mat.Rows(); i++){
      stream << "\t";
      for(uint j=0; j<mat.Cols(); j++)
	stream << std::setw(prec + 5) << mat[i][j] << " ";
      stream << std::endl;
    }
    
    return stream;
  }

}
