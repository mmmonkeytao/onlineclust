#include <sys/time.h>
#include <stdlib.h>
#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_Exception.hh"
#include "GPlib/GP_Histogram.hh"


using namespace std;
using namespace GPLIB;

GP_Histogram::GP_Histogram() : dim(0), nb_bins(), dyn_range(0),
			 bin_width(0), bins(), is_weighted(0), eps(1e-6) {}

GP_Histogram::GP_Histogram(uint new_nb_bins,
		     std::pair<double,double> const &range,
		     bool weighted, double empty_thresh) :
  dim(1), nb_bins(1, new_nb_bins), dyn_range(1),
  bin_width(1), bins(), is_weighted(weighted), eps(empty_thresh)
{
  dyn_range[0] = range;

  // Calculate bin width and bin borders
  if (is_weighted)
    bin_width[0] = (range.second - range.first) / (nb_bins[0] - 1);

  else
    bin_width[0] = (range.second - range.first) / nb_bins[0];

}

GP_Histogram::GP_Histogram(std::vector<uint> const &new_nb_bins,
		     std::vector<std::pair<double,double> > const &range,
		     bool weighted, double empty_thresh) :
  dim(range.size()), nb_bins(new_nb_bins), dyn_range(range),
  bin_width(range.size()), bins(), is_weighted(weighted), eps(empty_thresh)
{
  if(new_nb_bins.size() != range.size())
    throw GP_EXCEPTION("Number of bin dimensions must match range dimensions");

  std::vector<uint> idx(1);

  // Calculate bin width
  if (is_weighted)
    for(uint i=0; i<bin_width.size(); i++)
      bin_width[i] = (range[i].second - range[i].first) / (nb_bins[i] - 1);

  else
    for(uint i=0; i<bin_width.size(); i++)
      bin_width[i] = (range[i].second - range[i].first) / nb_bins[i];

}


GP_Histogram::GP_Histogram(GP_Histogram const &other) :
  dim(other.dim), nb_bins(other.nb_bins),
  dyn_range(other.dyn_range), bin_width(other.bin_width),
  bins(other.bins), is_weighted(other.is_weighted),
  eps(other.eps)
{}

std::vector<uint> GP_Histogram::GetBinIdx(std::vector<double> const &pix) const
{
  if (pix.size() != dim)
    throw GP_EXCEPTION("GP_Histogram dimension does not match.");

  for(uint k=0; k<dim; k++)
    if((pix[k] < dyn_range[k].first) || (pix[k] > dyn_range[k].second)){
      cout << pix[k] << " " << dyn_range[k].first << " " << dyn_range[k].second << " " << k << " " << dim << endl;
      throw GP_EXCEPTION("Pixel value is outside of histogram range.");
    }

  std::vector<uint> out(dim);
  
  for(uint k=0; k<dim; k++){

    out[k] = (uint)((pix[k] - dyn_range[k].first) / bin_width[k]);

    if(out[k] == nb_bins[k]) out[k]--;
  }

  return out;
}

std::vector<double> GP_Histogram::GetCoords(std::vector<uint> const &idx) const
{
  if (idx.size() != dim)
    throw GP_EXCEPTION("GP_Histogram dimension does not match.");

  for(uint k=0; k<dim; k++)
    if(idx[k] >= nb_bins[k]){
      cerr << "index: " << idx[k]<< endl;
      throw GP_EXCEPTION("Bin index is invalid.");
    }
  std::vector<double> coords(idx.size());

  // Richard: Add 0.5 here ?!
  for(uint k=0; k<idx.size(); k++)
    coords[k] = ((double)idx[k]) * bin_width[k] + dyn_range[k].first;

  return coords;
}

GP_Vector GP_Histogram::GetContent1D() const
{
  if(dim != 1)
    throw GP_EXCEPTION("GP_Histogram has not the correct dimension (should be 1).");

  GP_Vector out(nb_bins[0]);
  for(uint i=0; i<out.Size(); ++i)
    out[i] = (*this)[i];

  return out;
}

GP_Matrix GP_Histogram::GetContent2D() const
{
  if(dim != 2)
    throw GP_EXCEPTION("GP_Histogram has not the correct dimension (should be 2).");

  GP_Matrix out(nb_bins[0], nb_bins[1]);
  for(uint i=0; i<nb_bins[0]; ++i)
    for(uint j=0; j<nb_bins[1]; ++j){

      vector<uint> idx(2);
      idx[0] = i; idx[1] = j;
      out[i][j] = (*this)[idx];
    }

  return out;
}

void GP_Histogram::SetBinWidth(uint d, double width)
{
  if (d >= bin_width.size())
    throw GP_EXCEPTION("Invalid bin index.");

  bin_width[d] = width;
}

bool GP_Histogram::IsInRange(std::vector<double> const &coords) const
{
  for(uint k=0; k<coords.size(); k++)
    if((coords[k] < dyn_range[k].first) || (coords[k] > dyn_range[k].second))
      return false;
  return true;
}


double GP_Histogram::operator[](uint idx) const
{
  if(dim != 1)
    throw GP_EXCEPTION("Assignment operator of GP_Histogram called with a scalar index, but the "
		       "dimension is not 1.");

  return operator[](std::vector<uint>(1,idx));
}

double GP_Histogram::operator[](std::vector<uint> const &idx) const
{
  Bin_const_iterator it = bins.find(idx);

  if (it == bins.end())
    return 0.0;

  return it->second;
}

double &GP_Histogram::operator[](std::vector<uint> const &idx)
{
  return bins[idx];
}

const GP_Histogram &GP_Histogram::operator=(GP_Histogram const &other)
{
  if (this != &other){
    destroy();
    copy(other);
  }
  return (*this);
}

double GP_Histogram::Dist(GP_Histogram const &other) const
{
  Bin_const_iterator pos1, pos2;
  double dist = 0;
  int count = 0;

  pos1 = other.begin();
  pos2 = this->begin();

  while ((pos1 != other.end()) && (pos2 != this->end())){
    if (pos1->first > pos2->first){
      dist += pos2->second * pos2->second;
      pos2++;
    }
    else if (pos1->first < pos2->first){
      dist += pos1->second * pos1->second;
      pos1++;
    }
    else{
      dist += (pos2->second - pos1->second)*(pos2->second - pos1->second);
      pos1++;
      pos2++;
    }
    count++;
  }
  return sqrt(dist)/count;
}

bool GP_Histogram::Exists(std::vector<uint> const &idx) const
{
  return (bins.find(idx) != bins.end());
}

double GP_Histogram::Sum() const
{
  Bin_const_iterator it;
  double sum = 0;

  for(it = bins.begin(); it != bins.end(); it++)
    sum += it->second;

  return sum;
}

double GP_Histogram::Max(double cmp) const
{
  Bin_const_iterator it;
  double max = -HUGE_VAL;

  for(it = bins.begin(); it != bins.end(); it++)
    if (it->second > max &&
	it->second < cmp) max = it->second;
  return max;
}

std::vector<uint> GP_Histogram::ArgMax(double cmp) const
{
  Bin_const_iterator it, max_it;
  max_it = bins.begin();
  double max = -HUGE_VAL;

  for(it = bins.begin(); it != bins.end(); it++)
    if (it->second > max && 
	it->second < cmp){
      max_it = it;
      max = max_it->second;
    }
  return max_it->first;
}

double GP_Histogram::Min(double cmp) const
{
  Bin_const_iterator it;
  double min = HUGE_VAL;

  for(it = bins.begin(); it != bins.end(); it++)
    if (it->second < min &&
	it->second > cmp) min = it->second;
  return min;
}

std::vector<uint> GP_Histogram::ArgMin(double cmp) const
{
  Bin_const_iterator it, min_it;
  min_it = bins.begin();
  double min = HUGE_VAL;

  for(it = bins.begin(); it != bins.end(); it++)
    if (it->second < min && 
	it->second > cmp){
      min_it = it;
      min = min_it->second;
    }
  return min_it->first;
}

GP_Histogram GP_Histogram::operator+(GP_Histogram const &other) const
{
  if (this->nb_bins.size() != other.nb_bins.size())
    throw GP_EXCEPTION ("Number of bin dimensions do not match.");

  for(uint i=0; i<nb_bins.size(); ++i)
    if (this->GetNbBins(i) != other.GetNbBins(i))
      throw GP_EXCEPTION ("GP_Histogram sizes do not match.");

  GP_Histogram hist(*this);

  Bin_const_iterator it;
  for(it = other.begin(); it != other.end(); it++)
    hist[it->first] += it->second;

  return hist;
}

GP_Histogram &GP_Histogram::operator+=(GP_Histogram const &other)
{
  if (this->nb_bins.size() != other.nb_bins.size())
    throw GP_EXCEPTION ("Number of bin dimensions do not match.");

  for(uint i=0; i<nb_bins.size(); ++i)
    if (this->GetNbBins(i) != other.GetNbBins(i))
      throw GP_EXCEPTION ("GP_Histogram sizes do not match.");

  Bin_const_iterator it;
  for(it = other.begin(); it!= other.end(); it++)
    bins[it->first] += it->second;

  return *this;
}

GP_Histogram GP_Histogram::operator*(GP_Histogram const &other) const
{
  if (this->nb_bins.size() != other.nb_bins.size())
    throw GP_EXCEPTION ("Number of bin dimensions do not match.");

  for(uint i=0; i<nb_bins.size(); ++i)
    if (this->GetNbBins(i) != other.GetNbBins(i))
      throw GP_EXCEPTION ("GP_Histogram sizes do not match.");

  GP_Histogram hist(*this);

  Bin_const_iterator it;
  for(it = other.begin(); it != other.end(); it++)
    hist[it->first] = hist[it->first] * it->second;

  return hist;
}

GP_Histogram &GP_Histogram::operator*=(GP_Histogram const &other)
{
  if (this->nb_bins.size() != other.nb_bins.size())
    throw GP_EXCEPTION ("Number of bin dimensions do not match.");

  for(uint i=0; i<nb_bins.size(); ++i)
    if (this->GetNbBins(i) != other.GetNbBins(i))
      throw GP_EXCEPTION ("GP_Histogram sizes do not match.");

  Bin_const_iterator it;
  for(it = other.begin(); it!= other.end(); it++)
    bins[it->first] *= it->second;

  return *this;
}

GP_Histogram GP_Histogram::operator/(GP_Histogram const &other) const
{
  if (this->nb_bins.size() != other.nb_bins.size())
    throw GP_EXCEPTION ("Number of bin dimensions do not match.");

  for(uint i=0; i<nb_bins.size(); ++i)
    if (this->GetNbBins(i) != other.GetNbBins(i))
      throw GP_EXCEPTION ("GP_Histogram sizes do not match.");

  GP_Histogram hist(*this);

  Bin_const_iterator it;
  for(it = other.begin(); it != other.end(); it++)
    {
      // Richard 
      //cout << "(" << hist[it->first] << "/" << it->second << ")" << endl;
      hist[it->first] = hist[it->first] / it->second;
    }

  return hist;
}

double GP_Histogram::Corr(GP_Histogram const &other) const
{
  if (this->nb_bins.size() != other.nb_bins.size())
    throw GP_EXCEPTION ("Number of bin dimensions do not match.");

  for(uint i=0; i<nb_bins.size(); ++i)
    if (this->GetNbBins(i) != other.GetNbBins(i))
      throw GP_EXCEPTION ("GP_Histogram sizes do not match.");

  Bin_const_iterator it;
  double sum = 0;
  for(it = other.begin(); it!= other.end(); it++)
    sum += (*this)[it->first] * it->second;

  return sum;
}

double GP_Histogram::Entropy(double log_base) const
{
  double sum = Sum();
  double entropy = 0;

  Bin_const_iterator it;

  for(it = bins.begin(); it != bins.end(); it++){
    double prob = it->second / sum;
    if(prob > 1e-10 && prob < 1 - 1e-10)
      entropy += prob * log(prob);
  }
  entropy /= log(log_base);
  
  return -entropy;
}


void GP_Histogram::ScaleBins(double factor)
{
  std::vector<uint> idx;
  Bin_iterator it;
  for(it = bins.begin(); it != bins.end(); it++){
    idx = it->first;
    for(uint i=0; i<idx.size(); i++)
      idx[i] = (uint)(idx[i]*factor);
    const_cast<std::vector<uint>&>(it->first) = idx;
  }
}

void GP_Histogram::Crop(std::vector<uint> const &idx)
{
  Bin_iterator it;
  for(it = bins.begin(); it != bins.end(); it++)
    for(uint i=0; i<idx.size(); i++)
      if(it->first[i] > idx[i])
	bins.erase(it);
}



uint GP_Histogram::Increment(double entry, double val)
{
  if (dim > 1)
    throw GP_EXCEPTION("GP_Histogram is multidimensional. Method `Increment"
		       "must be called with a vector of entries");

  std::vector<double> entry_vec(1);
  entry_vec[0] = entry;


  std::vector<uint> idx = GetBinIdx(entry_vec);
  if(idx[0] >= nb_bins[0])
    idx[0] = nb_bins[0]-1;

  if (!is_weighted)
    bins[idx] += val;

  else{
    std::vector<double> lower_bound = GetCoords(idx);
    double frac = (entry_vec[0] - lower_bound[0]) / bin_width[0];
    bins[idx] += (1. - frac) * val;
    idx[0]++;
    if (idx[0] < nb_bins[0])
      bins[idx] += (frac * val);
  }

  return idx[0];
}

std::vector<uint> GP_Histogram::Increment(std::vector<double> const &entries, double val)
{
  if (entries.size() != dim)
    throw GP_EXCEPTION("Index vector length does not match histogram dimension" );

  std::vector<uint> idx = GetBinIdx(entries);
  for(uint i=0; i<idx.size(); i++)
    if(idx[i] >= nb_bins[i])
      idx[i] = nb_bins[i] - 1;

  if (!is_weighted)
    bins[idx] += val;

  else{

    std::vector<double> frac(dim);
    std::vector<uint> neighbor(dim);
    double value, factor;

    for(uint d=0; d<dim; d++){

      value   = entries[d] - dyn_range[d].first;
      idx[d]  = (uint)(value / bin_width[d]);
      frac[d] = fmod(value, bin_width[d]) / bin_width[d];
    }

    uint nb_neighbor_bins = 1 << dim;

    for(uint k=0; k<nb_neighbor_bins; k++){
      factor = 1;

      for(uint d=0; d<dim; d++)

	if (k & (1 << d)) {
	  neighbor[d] = idx[d] + 1;
	  factor *= frac[d];
	}
	else {
	  neighbor[d] = idx[d];
	  factor *= 1 - frac[d];
	}
      bins[neighbor] += (factor * val);
    }
  }
  
  return idx;
}

std::vector<uint> GP_Histogram::Increment(GP_Vector const &coords, double val)
{
  vector<double> std_vec(coords.Size());
  for(uint i=0; i<coords.Size(); ++i)
    std_vec[i] = coords[i];
    
  return Increment(std_vec, val);
}

void GP_Histogram::RemoveEmptyBins(double thresh)
{
  std::vector<std::vector<uint> > to_delete;

  for(Bin_iterator bin_it = bins.begin(); bin_it != bins.end(); bin_it++)
    if (bin_it->second < thresh)
      to_delete.push_back(bin_it->first);

  for(uint i=0; i<to_delete.size(); i++)
    bins.erase(to_delete[i]);
}

void GP_Histogram::WeightedSampling(uint nb_samples, 
				 std::vector<std::vector<double> > &samples) const
{
  double bin_sum = 0;
  double sample_bin;
  double smpl;
  std::vector<std::pair<double, Bin_const_iterator> > sample_bins;

  struct timeval tv;
  gettimeofday(&tv, NULL);

  srand(tv.tv_usec);

  for(Bin_const_iterator bin_it = bins.begin(); bin_it != bins.end(); bin_it++)
    sample_bins.push_back(make_pair(bin_sum += bin_it->second, bin_it));

  samples.clear();

  smpl = (double)rand()/ (double)RAND_MAX;
  double delta = smpl / (double) (nb_samples + 1);
  uint sample_bin_idx = 0;
  std::vector<double> sample(dim);

  for(uint i=0; i<nb_samples; i++){

    sample_bin = (delta + (double)i / (double) nb_samples) * bin_sum;

    while(sample_bin_idx < sample_bins.size() &&
	  (sample_bin > sample_bins[sample_bin_idx].first)) sample_bin_idx++;

    std::vector<uint> curr_bin = sample_bins[sample_bin_idx].second->first;

    for(uint d=0; d<dim; d++){
      smpl = (double)rand()/ (double)RAND_MAX;

      sample[d] = bin_width[d] * (smpl + (double) curr_bin[d]) + 
	dyn_range[d].first;
    }

    samples.push_back(sample);
  }
}

// Read GP_Histogram from binary file
bool GP_Histogram::Read(char const *filename)
{
  destroy();

  // Open input file
  ifstream hist_in( filename,ios::in|ios::binary);
  if ( ! hist_in ) {
    cerr << "Cannot open input file: " << filename << endl;
    exit(EXIT_FAILURE);
  }

  hist_in.read((char*)&is_weighted,     sizeof(bool));
  hist_in.read((char*)&dim,     sizeof(uint));

  nb_bins.resize(dim);
  for(uint i=0; i<dim; i++)
    hist_in.read((char*)&(nb_bins[i]), sizeof(uint));

  dyn_range.resize(dim);
  for(uint i=0; i<dim; i++){
    hist_in.read((char*)&(dyn_range[i].first),  sizeof(double));
    hist_in.read((char*)&(dyn_range[i].second), sizeof(double));
  }

  std::vector<uint> bin_idx(dim);
  double entry;

  while (!hist_in.eof()){

    for(uint k=0; k < dim; k++)
      hist_in.read((char*)&bin_idx[k], sizeof(uint));

    hist_in.read((char*)&entry, sizeof(double));
    bins[bin_idx] = entry;
  }

  hist_in.close();

  bin_width = std::vector<double>(dim);
  for(uint i=0; i<dim; i++)
    if (is_weighted)
      bin_width[i] = (dyn_range[i].second - dyn_range[i].first) / (nb_bins[i] - 1);
    else
      bin_width[i] = (dyn_range[i].second - dyn_range[i].first) / nb_bins[i];

  return 1;
}

// Write GP_Histogram into binary file
bool GP_Histogram::Write(char const *filename) const
{
  unsigned int l;

  // Open output file
  ofstream hist_out( filename, ios::out|ios::binary);
  if ( ! hist_out ) 
    throw GP_EXCEPTION2("Cannot open output file: %s", filename);

  hist_out.write((char*)&is_weighted, sizeof(bool));
  hist_out.write((char*)&dim,     sizeof(uint));
    
  for(uint i=0; i<nb_bins.size(); i++)
    hist_out.write((char*)&nb_bins[i], sizeof(uint));
    
  for(uint i=0; i<dyn_range.size(); i++){
    hist_out.write((char*)&dyn_range[i].first,  sizeof(double));
    hist_out.write((char*)&dyn_range[i].second, sizeof(double));
  }

  // Create buffer for histogram data
  Bin_const_iterator pos;
  for(pos = bins.begin(); pos != bins.end(); pos++){
    for (l =0; l < dim; l++)
      hist_out.write((char*)&(pos->first[l]), sizeof(uint));
    hist_out.write((char*)&pos->second, sizeof(double));
  }

  hist_out.close();

  return 1;
}

// Richard; Write the histogram to a vrml file
bool GP_Histogram::WriteVRML(char const *filename) const
{
  if(dim != 2) // currently this function only supports only 2d histograms
    {
      throw GP_EXCEPTION("Vrml output not supported for this number of dimensions dimension");
            
      return 0;
    }

  WRITE_FILE(file, filename);

  file << "VRML V2.0 utf8\n";
  file << "Shape {" << endl;
  file << "  appearance Appearance {" << endl;
  file << "  material Material {}" << endl;
  file << "  }" << endl;
  file << "geometry ElevationGrid {" << endl;
  file << "  xDimension   " << nb_bins[0] << endl;
  file << "  zDimension   " << nb_bins[1] << endl;
  file << "  xSpacing   1" << endl;
  file << "  zSpacing   1" << endl;
  file << "  solid    FALSE" << endl;
  file << "  creaseAngle   0" << endl;
  file << "  height [";

  double max = Max();
  uint count = 8;

  // add heights to vrml file
  for (uint x=nb_bins[0]; 0 <  x;x--)
    {
      for (uint y=0;y < nb_bins[1];y++)
	{
	  if (count > 7)
	    {
	      count = 0;
	      file << "\n          ";
	    }
	  std::vector<uint> Coord(2);
	  Coord[0] = x;
	  Coord[1] = y;

	  if (Exists(Coord))
	    {
	      // todo: Richard, fix this ?!
	      float scale = 10;
	      float cur = log(exp((double)1) + (*this)[Coord]);
	      float lmax = log(exp((double)1) +  max);
	      file << scale*cur/lmax << ", ";
	    }
	  else
	    file << "0.0, ";

	  count++;
	}
    }
  file << "]" << endl;

  // add colors to vrml file
  file << "  colorPerVertex TRUE" << endl;
  file << "     color Color  {" << endl;
  file << "        color [" << endl;
  for (uint x=nb_bins[0]; 0 <  x;x--)
    {
      for (uint y=0;y < nb_bins[1];y++)
	{
	  std::vector<uint> Coord(2);
	  Coord[0] = x;
	  Coord[1] = y;
	  if (Exists(Coord))
	    {
	      // todo: Richard, fix this ?!
	      float cur = log(exp((double)1) + (*this)[Coord]);
	      float lmax = log(exp((double)1) +  max);
	      float clr = 1.0 - cur/lmax;
	      if (clr < 0)
		clr = 0;
	      file << "     " << clr << " " << clr <<  " " << clr << endl;
	    }
	  else
	    file << "     0.0 0.0 1.0 ," << endl;
	}
    }
  file << "     ]}" << endl;
  file << "  }}" << endl;
  file.close();
  return 1;
}

bool GP_Histogram::WritePPM(char const *filename,int resX) const
{
  if(dim != 1) // currently this function only supports only 2d histograms
    throw GP_EXCEPTION("Ppm output not supported for this number of dimensions dimension");

  int resY = (int)((double)resX / 4.0 * 3.0);

  // get the maxium/minimum range from the first dimension
  double minrange = dyn_range[0].first;
  double maxrange = dyn_range[0].second;

  // get the maximum 'height' from the bins
  double binmaxvalue = Max();

  //write the file
  WRITE_FILE(file,filename);
  file << "P6\n" << resX << " " << resY << "\n255\n";

  for(int iY = resY - 1; 0 <= iY; iY--)
    for(int iX = 0; iX < resX; iX++)
      {
	char color = 255; // or current height

	std::vector<double> Coord(1);
	std::vector<uint> BinIdx(1);
	// calculate the bin which corresponds to current X coordinate
	Coord[0] = (double)iX / (double)resX * (maxrange - minrange) + minrange;
	BinIdx = GetBinIdx(Coord);

	// calc the bin height
	if (Exists(BinIdx))
	  if ((*this)[BinIdx] / binmaxvalue > (double)iY / (double)resY)
	    color = 1;

	file << color << color << color;
      }
  return true;
}


bool GP_Histogram::WriteTextFile(char const *filename,int dimIdx) const
{
  WRITE_FILE(file,filename);

  GP_Histogram::Bin_const_iterator pos;
  std::vector<double> coords;

  for(pos = begin(); pos != end(); ++pos)
    {
      // set precision
      int oldpres = file.precision();
      file.precision(3);

      coords = GetCoords(pos->first);
      if (dimIdx < 0)
	for (uint k = 0; k < pos->first.size(); k++)
	  file << coords[k] << " ";
      else
	file << coords[dimIdx] << " ";

      file.precision(oldpres);
      file << pos->second << endl;
    }
  return true;
}

void GP_Histogram::Normalize(bool div_by_area)
{
  double sum = this->Sum();
  
  if(fabs(sum) < EPSILON)
    return;

  Bin_iterator it = bins.begin();

  double bin_area = 1;
  for(uint i=0; i<bin_width.size(); i++)
    bin_area *= bin_width[i];

  if(div_by_area && fabs(bin_area) < EPSILON)
    return;

  while(it != bins.end()){
    if(div_by_area)
      it->second /= (sum * bin_area);
    else
      it->second /= sum;
    it++;
  }
}

void GP_Histogram::UpdateRange(uint d, double min, double max)
{
  if (d >= dim)
    throw GP_EXCEPTION("Invalid index in GP_Histogram dimension.");
  dyn_range[d] = make_pair(min, max);
}

void GP_Histogram::copy(GP_Histogram const &other)
{
  dim         = other.dim;
  nb_bins     = other.nb_bins;
  dyn_range   = other.dyn_range;
  bin_width   = other.bin_width;
  bins        = other.bins;
  is_weighted = other.is_weighted;
  eps         = other.eps;
}

void GP_Histogram::destroy()
{
  dim = 0;
  nb_bins.clear();
  bins.clear();
}

// Overloaded << operator:

ostream &operator<<(ostream &stream, GP_Histogram const &hist)
{
  std::vector<int> idx(hist.GetDim());
  GP_Histogram::Bin_const_iterator pos;
  uint k;

  std::vector<double> coords;

  for(pos = hist.begin(); pos != hist.end(); ++pos){
    for (k = 0; k < pos->first.size(); k++)
      stream << pos->first[k] << " ";
    // Richard: added 'world coordinate' output

    // set precision
    int oldpres = stream.precision();
    stream.precision(3);

    stream << "(" << flush;
    coords = hist.GetCoords(pos->first);
    for (k = 0; k < pos->first.size(); k++)
      {
	stream << coords[k];
	if (k < pos->first.size() - 1)
	  stream << ",";
      }

    stream.precision(oldpres);
    stream << ")\t" << pos->second << endl;
  }

  stream << "\0";
  return stream;
}


