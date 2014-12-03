#ifndef GP_VECTOR_HH
#define GP_VECTOR_HH

#include <vector>
#include <iostream>
#include <sys/types.h>

namespace GPLIB {


  /*!
   * A vector class
   *
   * We use this vector class only internally, and it only provides the 
   * minimal functionality needed for GPlib.
   */
  class GP_Vector
  {
  public:

    /*!
     * Default constructor
     */
    GP_Vector() : _data() {}

    /*!
     * Makes a zero-vector with a given length and initialization value
     */
    GP_Vector(uint len, double init = 0.) : _data(len, init) {}

    /*!
     * Builds a vector from an STL vector
     */
    GP_Vector(std::vector<double> const &stlvec) : _data(stlvec) {}

    /*!
     * Returns a pointer to the data
     */
    const double *Data() const {return _data.data(); }

    /*!
     * Returns a pointer to the data
     */
    double *Data() {return _data.data(); }

    /*!
     * Returns the stl vector of the data
     */
    operator std::vector<double> const &() const {return _data; }

    /*!
     * Casts the vector into a scalar. The vector must have only one element.
     */
    operator double const &() const;
   
    /*!
     * Returns the length of the vector;
     */
    uint Size() const { return _data.size(); } 

    /*!
     * Changes the length of the vector
     */
    void Resize(uint n) { _data.resize(n); }

    /*!
     * Get a subvector
     */
    GP_Vector SubVector(uint begin, uint end) const;

    /*!
     * Same as Size(), for compatibility
     */
    uint size() const { return _data.size(); } 

    bool AnyIsNaN() const;
    bool AllIsNaN() const;
    bool AnyIsInf() const;
    bool AllIsInf() const;

    /*!
     * Index operator
     */
    double operator[](uint i) const;

    /*!
     * Index operator
     */
    double &operator[](uint i);

    /*!
     * Vector-scalar multiplication
     */
    GP_Vector operator*(double x) const;

    /*!
     * Vector-scalar division
     */
    GP_Vector operator/(double x) const;

    /*!
     * Vector-scalar multiplication
     */
    GP_Vector const &operator*=(double x);

    /*!
     * Vector-scalar division
     */
    GP_Vector const &operator/=(double x);

    /*!
     * Elementwise arithmetics
     */
    GP_Vector operator+(GP_Vector const &other) const;

    /*!
     * Sign change
     */
    GP_Vector operator-() const;

    /*!
     * Elementwise arithmetics
     */
    GP_Vector operator-(double val) const;

    /*!
     * Elementwise arithmetics
     */
    GP_Vector operator-(GP_Vector const &other) const;

    /*!
     * Elementwise arithmetics
     */
    GP_Vector operator*(GP_Vector const &other) const;

    /*!
     * Elementwise arithmetics
     */
    GP_Vector operator/(GP_Vector const &other) const;

    /*!
     * Elementwise arithmetics
     */
    GP_Vector const &operator+=(GP_Vector const &other);

    /*!
     * Elementwise arithmetics
     */
    GP_Vector const &operator-=(GP_Vector const &other);

    /*!
     * Elementwise arithmetics
     */
    GP_Vector const &operator*=(GP_Vector const &other);

    /*!
     * Elementwise arithmetics
     */
    GP_Vector const &operator/=(GP_Vector const &other);

    /*!
     * The sum of all elements
     */
    double Sum() const;
    
    /*!
     * The product of all elements
     */
    double Prod() const;

    /*!
     * The dot product o two vectors
     */
    double Dot(GP_Vector const &other) const;
    double dot(GP_Vector const &other) const
    { return Dot(other); }

    /*!
     * The Euclidean norm
     */
    double Norm() const;

    double NormSqr() const;

    /*!
     * Returns the maximum value of all entries
     */
    double Max() const;

    /*!
     * Returns the index that refers to the maximum value of all entries
     */
    uint ArgMax() const;

    /*!
     * Computes the absolute value
     */
    GP_Vector Abs() const;

    /*!
     * Computes the square of each entry
     */
    GP_Vector Sqr() const;

    /*!
     * Computes the exponential of each entry
     */
    GP_Vector Exp() const;

    /*!
     * Computes the logarithm of each entry
     */
    GP_Vector Log() const;

    /*!
     * Appends a new value at the end
     */
    void Append(double x);
    
    /*!
     * Appends another vector at the end
     */
    void Append(GP_Vector const &other);

    /*!
     * Removes the last element
     */
    void RemoveLast();

    /*!
     * Reads vector from an ASCII file at a given position
     * Returns the new file position
     */
    int Read(std::string filename, int pos = 0);

    /*!
     * Writes vector into an ASCII file
     */
    void Write(std::string filename) const;

  private:

    std::vector<double> _data;
  };

  /*!
   * Scalar-vector multiplication
   */
  GP_Vector operator* (double x, GP_Vector const &m);
  
  /*!
   * Scalar-vector division
   */
  GP_Vector operator/ (double x, GP_Vector const &m);

  /*!
   * Writes a vector into an ostream
   */
  std::ostream &operator<<(std::ostream &stream, GP_Vector const &vec);

  /*!
   * Reads a vector from an istream; the length of the vector must be
   * specified beforehand
   */
  std::istream &operator>>(std::istream &stream, GP_Vector &vec);

}




#endif
