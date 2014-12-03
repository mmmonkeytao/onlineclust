#ifndef GP_HISTOGRAM_HH
#define GP_HISTOGRAM_HH

#include <sys/types.h>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include "GPlib/GP_Vector.hh"
#include "GPlib/GP_Matrix.hh"

namespace GPLIB {
  /*!
   * \class GP_Histogram
   *
   * Definition of class Histogram
   */

  class GP_Histogram
  {
  public:

    typedef std::map<std::vector<uint>, double>   Bin_type;
    typedef Bin_type::iterator          Bin_iterator;
    typedef Bin_type::const_iterator    Bin_const_iterator;

    // The constructors
    GP_Histogram();

    /*!
     * The constructor
     *
     * Constructs a 1-d histogram with a given number of bins
     * and a value range (given as a pair (min, max));
     * A threshold can be set for bins that are considered empty
     * (this is only for weighted hisograms).
     */
    GP_Histogram(uint nb_bins, std::pair<double,double> const &range,
		 bool weighted = false, double empty_thresh = 1e-6);

    /*!
     * The constructor
     *
     * Constructs a n-d histogram with a given number of bins
     * and a set of value ranges (given as pairs (min, max));
     * The dimension of the histogram is given by the length
     * of the vector `range`.
     * A threshold can be set for bins that are considered empty
     * (this is only for weighted histograms).
     */
    GP_Histogram(std::vector<uint> const &nb_bins, 
		 std::vector<std::pair<double,double> > const &range,
		 bool weighted = 0, double empty_thresh = 1e-6);

    GP_Histogram(GP_Histogram const &other);

    // Read histogram data
    unsigned int GetDim()     const { return dim; }
    unsigned int GetNbBins(uint i)  const { return nb_bins[i];}
    Bin_type const &GetBins() const { return bins;};
    Bin_type &GetBins() { return bins;};
    bool IsWeighted()         const { return is_weighted; }
    bool IsInRange(std::vector<double> const &coords) const;

    double operator[](uint idx) const;
    double operator[](std::vector<uint> const &idx) const;
    double &operator[](std::vector<uint> const &idx);

    std::vector<std::pair<double,double> > const &GetRange() const { return dyn_range; };
    std::vector<uint> GetBinIdx(std::vector<double> const &coords) const;
    double GetBinWidth(uint d) const { return bin_width[d]; }
    std::vector<double> GetCoords(std::vector<uint> const &idx) const;
    /*!
     * Returns the content of all bins as a Vector. Only valid if the dimension is 1
     */
    GPLIB::GP_Vector GetContent1D() const;

    /*!
     * Returns the content of all bins as a Matrix. Only valid if the dimension is 2
     */
    GPLIB::GP_Matrix GetContent2D() const;

    void SetBinWidth(uint d, double width);

    // Operators
    GP_Histogram operator+(GP_Histogram const &other) const;
    GP_Histogram &operator+=(GP_Histogram const &other);
    GP_Histogram operator*(GP_Histogram const &other) const;
    GP_Histogram &operator*=(GP_Histogram const &other);
    GP_Histogram operator/(GP_Histogram const &other) const;

    bool Exists(std::vector<uint> const&) const;
    double Sum() const;
    double Max(double comp = HUGE_VAL) const;
    std::vector<uint> ArgMax(double comp = HUGE_VAL) const;
    double Min(double comp = -HUGE_VAL) const;
    std::vector<uint> ArgMin(double comp = -HUGE_VAL) const;
    double Corr(GP_Histogram const &other) const;
    double Entropy(double log_base = 10.) const;

    void ScaleBins(double factor);
    void Crop(std::vector<uint> const &idx);

    /*!
     * increments the bin where 'entry' falls in; returns the bin index
     */
    uint Increment(double entry, double val = 1);

    /*!
     * increments the bin where 'entry' falls in; returns the bin index
     */
    std::vector<uint> Increment(std::vector<double> const &coords, double val = 1);
    std::vector<uint> Increment(GPLIB::GP_Vector const &coords, double val = 1);

    /*!
     * increments the bin with index 'idx'
     */
    void Increment(std::vector<uint> const &idx, double val = 1) { bins[idx] += val; }

    void RemoveEmptyBins(double thresh);

    void WeightedSampling(uint nb_samples, std::vector<std::vector<double> > &samples) const;

    /*!
     * Get begin and end of bin vector
     */
    Bin_const_iterator begin() const { return bins.begin(); }

    /*!
     * Get begin and end of bin vector
     */
    Bin_const_iterator end()   const { return bins.end(); }


    /*!
     * Get begin and end of bin vector
     */
    Bin_iterator begin() { return bins.begin(); }

    /*!
     * Get begin and end of bin vector
     */
    Bin_iterator end()  { return bins.end(); }

    /*!
     * Assignment operator
     */
    GP_Histogram const &operator=(GP_Histogram const&);

    /*!
     * Distance calculation
     */
    double Dist(GP_Histogram const &) const;

    //! Read histogram from binary file
    bool Read(char const *filename);

    //! Write histogram into binary file
    bool Write(char const *filename) const;

    //! Write histogram into Vrml file
    bool WriteVRML(char const *filename) const;

    bool WritePPM(char const *filename,int resX = 512) const;

    //! Writes the histogram into a txt file, used for parsing it to gnuplot
    bool WriteTextFile(char const *filename,int dimIdx = -1) const;

    //! Normalization. If div_by_vol is true, all bins are divided by the product 
    // of bin sum and total bin volume, otherwise only by the bin sum
    void Normalize(bool div_by_vol = false);

  protected:

    /*!
     * Changes the dynamic range in dimension 'd' to 'min' -- 'max'
     */
    void UpdateRange(uint d, double min, double max);

  private:
    uint dim;
    std::vector<uint> nb_bins;
    std::vector<std::pair<double,double> > dyn_range;
    std::vector<double> bin_width;
    Bin_type bins;
    bool is_weighted;
    double eps;

    void copy(GP_Histogram const&);
    void destroy();
  };

  /*
   *
   * Overloaded << operator:
   *
   */

  std::ostream &operator<<(std::ostream &, GP_Histogram const &);

}
#endif
