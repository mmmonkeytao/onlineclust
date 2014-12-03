#ifndef GP_MATRIX_HH
#define GP_MATRIX_HH

#include <iostream>
#include <gsl/gsl_linalg.h>
#include "GPlib/GP_Vector.hh"

namespace GPLIB {


  /*!
   * A matrix class
   *
   * We use this matrix class only internally, and it only provides the 
   * minimal functionality needed for GPlib.
   */
  class GP_Matrix
  {
  public:

    /*!
     * Default constructor
     */
    GP_Matrix() : _data() {}

    /*!
     * Builds a matrix of a given size and initialization value
     */
    GP_Matrix(uint rows, uint cols, double init = 0.) : 
      _data(rows, std::vector<double>(cols, init)) {}

    /*!
     * Builds a matrix with one row or column from a vector of numbers. 
     * If 'col_vecs' is true, the GP_Vectors are interpreted as column 
     */
    explicit GP_Matrix(std::vector<double> const &vectors, 
		       bool col_vec = true) :
      _data(col_vec ? vectors.size() : 1,
	    col_vec ? std::vector<double>(1) : 
	    std::vector<double>(vectors.size()))
    {
      for(uint i=0; i<_data.size(); ++i)
	for(uint j=0; j<_data[i].size(); ++j){
	  if(col_vec)
	    _data[i][j] = vectors[i];
	  else
	    _data[i][j] = vectors[j];
	}
    }

    /*!
     * Builds a matrix from a vector of GP_Vectors. If 'col_vecs' is true,
     * the GP_Vectors are interpreted as column 
     */
    GP_Matrix(std::vector<GP_Vector> const &vectors, bool col_vecs = true) :
      _data(col_vecs ? (vectors.size() ? vectors[0].Size() : 0) :
	    vectors.size(),
	    col_vecs ? std::vector<double>(vectors.size()) :
	    (vectors.size() ? std::vector<double>(vectors[0].Size()) : 
	     std::vector<double>()) ) 
    {
      for(uint i=0; i<_data.size(); ++i)
	for(uint j=0; j<_data[i].size(); ++j){
	  if(col_vecs)
	    _data[i][j] = vectors[j][i];
	  else
	    _data[i][j] = vectors[i][j];
	}
    }

    /*!
     * Returns number of rows
     */
    uint Rows() const;

    /*!
     * Returns number of columns
     */
    uint Cols() const;

    /*!
     * Returns the row with the given index
     */
    GP_Vector Row(uint idx) const;

    /*!
     * Returns the column with the given index
     */
    GP_Vector Col(uint idx) const;

    /*!
     * Converts the matrix into a GSL matrix
     */
    gsl_matrix *MakeGSLMatrix() const;

    /*!
     * Returns the transpose of the matrix
     */
    GP_Matrix Transp() const;

    /*!
     * Returns the specified submatrix 
     */
    GP_Matrix SubMatrix(uint x1, uint y1, uint x2, uint y2) const;

    /*!
     * Index operator
     */
    std::vector<double> const &operator[](uint i) const;

    /*!
     * Index operator
     */
    std::vector<double> &operator[](uint i);

    /*!
     * Elementwise arithmetics
     */
    GP_Matrix operator+(GP_Matrix const &other) const;

    /*!
     * Elementwise arithmetics
     */
    GP_Matrix operator-(GP_Matrix const &other) const;

    /*!
     * Elementwise arithmetics in-place, returns the result
     */
    GP_Matrix &operator+=(GP_Matrix const &other);

    /*!
     * Elementwise arithmetics in-place, returns the result
     */
    GP_Matrix &operator-=(GP_Matrix const &other);

    /*!
     * Elementwise arithmetics
     */
    GP_Matrix ElemMult(GP_Matrix const &other) const;

    /*!
     * Elementwise arithmetics
     */
    GP_Matrix ElemDiv(GP_Matrix const &other) const;

    /*!
     * Matrix-scalar multiplication
     */
    GP_Matrix operator* (double x) const;

    /*!
     * Matrix-vector multiplication
     */
    GP_Vector operator*(GP_Vector const &v) const;

    /*!
     * Matrix-matrix multiplication
     */
    GP_Matrix operator*(GP_Matrix const &M) const;

    /*!
     * Matrix-scalar division
     */
    GP_Matrix operator/ (double x) const;

    /*!
     * Computes M^T * A, where M is this matrix
     */
    GP_Matrix TranspTimes(GP_Matrix const &A)  const;

    /*!
     * Computes M * A^T, where M is this matrix
     */
    GP_Matrix TimesTransp(GP_Matrix const &A)  const;

    /*!
     * Returns the main diagonal
     */
    GP_Vector Diag() const;

    /*!
     * Inverts the matrix and returns it
     */
    GP_Matrix const &Invert();

    /*!
     * Returns the inverse of this matrix
     */
    GP_Matrix Inverse() const;

    /*!
     * Returns the inverse of this matrix using the Cholesky decomposition.
     * This is numerically more stable, but works only for symmetric, 
     * positive-semidefinite matrices
     */
    GP_Matrix InverseByChol() const;

    /*!
     * The Frobenius norm
     */
    double Fnorm() const;

    /*!
     * Makes a diagonal matrix from a vector
     */
    static GP_Matrix Diag(GP_Vector const &elems);

    /*!
     * Makes an n x n identity matrix
     */
    static GP_Matrix Identity(uint n);

    /*!
     * Returns the sum of all elements
     */
    double Sum() const;

    /*!
     * Returns the sum of the diagonal elements
     */
    double Trace() const;

    /*!
     * Returns the sum of the log of the diagonal elements
     */
    double LogTrace() const;

    /*!
     * Returns the maximum of all elements
     */
    double Max() const;

    /*!
     * Returns the minimum of all elements
     */
    double Min() const;

    /*!
     * Computes the outer product of the vector with itself
     */
    static GP_Matrix OutProd(GP_Vector const &v);

    /*!
     * Computes the sum of the logarithm of the diagonal
     */
    double SumLogDiag() const;

    /*!
     * Computes the determinant
     */
    double Det() const;

    /*!
     * Computes the absolute value for all entries
     */
    GP_Matrix Abs() const;

    /*!
     * Computes the singular value decomposition
     */
    void SVD(GP_Matrix &U, GP_Matrix &D, GP_Matrix &VT) const;

    /*!
     * Cholesky decomposition in-place. 
     * Returns false if matrix is not positive semi-definite.
     */
    bool Cholesky();

    /*! 
     * Compute Cholesky decomp of Matrix B = I + W^0.5 * K * W^0.5 
     * W is a diagonal matrix and is represented only as a vector
     */
    void CholeskyDecompB(GP_Vector const &w_sqrt);

    /*!
     * Solves L * x = b where L is supposed to be lower triangular
     */
    GP_Vector ForwSubst(GP_Vector const &b) const;
    GP_Matrix ForwSubst(GP_Matrix const &B) const;

    /*!
     * Solves L * x = b where L is supposed to be upper triangular
     */
    GP_Matrix TForwSubst(GP_Matrix const &B) const;

    /*!
     * Solves L^T * x = b where L is supposed to be lower triangular
     */
    GP_Vector BackwSubst(GP_Vector const &b) const;
    GP_Matrix BackwSubst(GP_Matrix const &B) const;

    /*!
     * Solves A * x = b using Cholesky decomp b
     */
    GP_Vector SolveChol(GP_Vector const &b) const;
    GP_Matrix SolveChol(GP_Matrix const &B) const;

    
    /*!
     * Computes eigen vectors and eigen values. Assumes that the matrix is 
     * symmetric. 
     */
    void EigenSolveSymm(GP_Vector &eval, GP_Matrix &evec, 
			bool sortflag = true) const;

    /*!
     * Returns the sum of all rows
     */
    GP_Vector RowSum() const;

    /*!
     * Add a new row at the lower end of the matrix
     */
    void AppendRow(GP_Vector const &row);

    /*!
     * Add a new column at the right end of the matrix
     */
    void AppendColumn(GP_Vector const &col);

    /*!
     * Appends the other matrix to the right of this matrix. 
     * Both matrices must have the same number of rows.
     */
    void AppendVert(GP_Matrix const &other);

    /*!
     * Appends the other matrix to the bottom of this matrix. 
     * Both matrices must have the same number of columns.
     */
    void AppendHor(GP_Matrix const &other);

    /*!
     * Removes the last row
     */
    void RemoveLastRow();

    /*!
     * Removes the last column
     */
    void RemoveLastColumn();

    /*!
     * Removes the row and the column with the specified index
     * and returns the smaller matrix
     */
    GP_Matrix RemoveRowAndColumn(uint idx) const;

    /*!
     * Reads matrix from an ASCII file from a given position,
     * returns the new file position
     */
    int Read(std::string filename, int pos = 0);

    /*!
     * Writes matrix into an ASCII file
     */
    void Write(std::string filename) const;

    /*!
     * Writes matrix into a.pgm file
     */
    void ExportPGM(std::string filename) const;

    static GP_Matrix Kronecker(GP_Matrix const &other, uint m, uint n);

    GP_Matrix Kronecker(GP_Matrix const &other) const;


  private:

    std::vector<std::vector<double> > _data;

  };

  /*!
   * Scalar-matrix multiplication
   */
  GP_Matrix operator*(double x, GP_Matrix const &m);

  /*!
   * Scalar-matrix division
   */
  GP_Matrix operator/(double x, GP_Matrix const &m);
  
  /*!
   * Vector-matrix multiplication (vector is assumed to be a row-vector)
   */
  GP_Vector operator* (GP_Vector const &vec, GP_Matrix const &mat);

  /*!
   * Writes a matrix into an ostream
   */
  std::ostream &operator<<(std::ostream &stream, GP_Matrix const &mat);
}




#endif
