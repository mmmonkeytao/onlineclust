#ifndef GP_TENSOR_HH
#define GP_TENSOR_HH

#include <iostream>
#include <gsl/gsl_linalg.h>
#include "GPlib/GP_Matrix.hh"

namespace GPLIB {

  class all
  {
    void operator()() const {}
  };

  /*!
   * A tensor class with 3 dimensions
   *
   */
  class GP_Tensor
  {
  public:


    /*!
     * Default constructor
     */
    GP_Tensor() : _data() {}

    /*!
     * Builds a matrix of a given size and initialization value
     */
    GP_Tensor(uint rows, uint cols, uint layers, double init = 0.) : 
      _data(rows, std::vector<std::vector<double> >(cols, std::vector<double>(layers, init))) {}


    /*!
     * Returns number of rows
     */
    uint Rows() const;

    /*!
     * Returns number of columns
     */
    uint Cols() const;

    /*!
     * Returns number of layers
     */
    uint Layers() const;

    /*!
     * Returns the row with the given index
     */
    GP_Matrix Row(uint idx) const;

    /*!
     * Returns the column with the given index
     */
    GP_Matrix Col(uint idx) const;

    /*!
     * Returns the layer with the given index
     */
    GP_Matrix Layer(uint idx) const;

    GP_Vector operator()(uint i, uint j, all) const;

    GP_Vector operator()(uint i, all, uint k) const;

    GP_Vector operator()(all, uint j, uint k) const;

    /*!
     * Index operator
     */
    std::vector<std::vector<double> > const &operator[](uint i) const;

    /*!
     * Index operator
     */
    std::vector<std::vector<double> > &operator[](uint i);


  private:

    std::vector<std::vector<std::vector<double> > > _data;
  };

}

#endif
