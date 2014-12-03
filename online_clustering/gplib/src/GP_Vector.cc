#include "GPlib/GP_Vector.hh"
#include "GPlib/GP_Constants.hh"
#include "GPlib/GP_Exception.hh"

#include <iomanip>
#include <fstream>
#include <math.h>

namespace GPLIB {

  GP_Vector GP_Vector::SubVector(uint begin, uint end) const
  {
    if(begin >= _data.size() || end > _data.size() || begin > end)
      throw GP_EXCEPTION("Invalid bounds in SubVector");

    GP_Vector out;
    out._data = std::vector<double>(end - begin);
    for(uint i=begin; i<end; ++i)
      out._data[i-begin]  = _data[i];

    return out;
  }

  GP_Vector::operator double const &() const 
  {
    if(_data.size() != 1)
      throw GP_EXCEPTION("Invalid data type. Must be a scalar");
    return _data[0]; 
  }


  bool GP_Vector::AnyIsNaN() const
  {
    for(uint i=0; i<_data.size(); ++i)
      if(isnan(_data[i]))
	return true;
    return false;
  }
  
  bool GP_Vector::AllIsNaN() const
  {
    for(uint i=0; i<_data.size(); ++i)
      if(!isnan(_data[i]))
	return false;
    return true;
  }
  
  bool GP_Vector::AnyIsInf() const
  {
    for(uint i=0; i<_data.size(); ++i)
      if(isinf(_data[i]))
	return true;
    return false;
  }

  bool GP_Vector::AllIsInf() const
  {
    for(uint i=0; i<_data.size(); ++i)
      if(!isinf(_data[i]))
	return false;
    return true;
  }
  
  double GP_Vector::operator[](uint i) const
  {
    if(i >= _data.size())
      throw GP_EXCEPTION2("Invalid index %d into vector", i);
    return _data[i];
  }

  double &GP_Vector::operator[](uint i)
  {
    if(i >= _data.size())
      throw GP_EXCEPTION2("Invalid index %d into vector", i);
    return _data[i];
  }

  GP_Vector GP_Vector::operator*(double x) const
  {
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] * x;
    return out;
  }
  
  GP_Vector GP_Vector::operator/(double x) const
  {
    if(fabs(x) < 1e-15)
      throw GP_EXCEPTION("Division by zero.");

    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] / x;
    return out;    
  }

  GP_Vector const &GP_Vector::operator*=(double x)
  {
    for(uint i=0; i<_data.size(); ++i)
      _data[i] *= x;
    return *this;
  }
  
  GP_Vector const &GP_Vector::operator/=(double x)
  {
    if(fabs(x) < 1e-15)
      throw GP_EXCEPTION("Division by zero.");

    for(uint i=0; i<_data.size(); ++i)
      _data[i] /= x;
    return *this;    
  }

  GP_Vector GP_Vector::operator+(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }
    
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] + other[i];
    return out;
  }
  
  GP_Vector GP_Vector::operator-() const
  {
    GP_Vector out(*this);
    for(uint i=0; i<out._data.size(); ++i)
      out[i] = -out[i];

    return out;
  }

  GP_Vector GP_Vector::operator-(double val) const
  {
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] - val;
    return out;  
  }

  GP_Vector GP_Vector::operator-(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] - other[i];
    return out;  
  }
  
  GP_Vector GP_Vector::operator*(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }
  
    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] * other[i];
    return out;
  }
  
  GP_Vector GP_Vector::operator/(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    GP_Vector out(_data.size());
    for(uint i=0; i<_data.size(); ++i)
      out[i] = _data[i] / other[i];
    return out;
  }

  GP_Vector const &GP_Vector::operator+=(GP_Vector const &other)
  {
    if(_data.size() == 0)
      return other;

    else if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }
    
    for(uint i=0; i<_data.size(); ++i)
      _data[i] += other[i];
    return *this;
  }
  
  GP_Vector const &GP_Vector::operator-=(GP_Vector const &other)
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    for(uint i=0; i<_data.size(); ++i)
      _data[i] -= other[i];
    return *this;  
  }
  
  GP_Vector const &GP_Vector::operator*=(GP_Vector const &other) 
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }
  
    for(uint i=0; i<_data.size(); ++i)
      _data[i] *= other[i];

    return *this;
  }
  
  GP_Vector const &GP_Vector::operator/=(GP_Vector const &other)
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    for(uint i=0; i<_data.size(); ++i)
      _data[i] /=  other[i];

    return *this;
  }

  double GP_Vector::Sum() const
  {
    double sum = 0;
    for(uint i=0; i<_data.size(); ++i)
      sum += _data[i];

    return sum;
  }
  
  double GP_Vector::Prod() const
  {
    double prod = 1;
    for(uint i=0; i<_data.size(); ++i)
      prod *= _data[i];

    return prod;
  }
  
  double GP_Vector::Dot(GP_Vector const &other) const
  {
    if(other.Size() != _data.size()){
      std::stringstream msg;
      msg << _data.size() << " != " << other.Size();
      throw GP_EXCEPTION2("Vector dimensions do not match: %s", msg.str());
    }

    double sum = 0;
    for(uint i=0; i<_data.size(); ++i)
      sum += _data[i] * other[i];

    return sum;
  }
  
  double GP_Vector::Norm() const
  {
    return sqrt(Dot(*this));
  }

  double GP_Vector::NormSqr() const
  {
    return Dot(*this);
  }

  GP_Vector GP_Vector::Abs() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = fabs(_data[i]);
    
    return out;
  }

  double GP_Vector::Max() const
  {
    if(_data.size() == 0)
      return -HUGE_VAL;

    double m = _data[0];

    for(uint i=0; i<_data.size(); i++)
      if(_data[i] > m)
	m = _data[i];
    
    return m;
  }

  uint GP_Vector::ArgMax() const
  {
    if(_data.size() == 0)
      return 0;

    double max = _data[0];
    uint argmax = 0;

    for(uint i=0; i<_data.size(); i++)
      if(_data[i] > max){
	max = _data[i];
	argmax = i;
      }
    
    return argmax;
  }
  
  GP_Vector GP_Vector::Sqr() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = _data[i] * _data[i];
    
    return out;
  }
 
  GP_Vector GP_Vector::Exp() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = exp(_data[i]);
    
    return out;
  }
 
  GP_Vector GP_Vector::Log() const
  {
    GP_Vector out(_data.size());
    
    for(uint i=0; i<_data.size(); i++)
      out[i] = log(_data[i]);
    
    return out;
  }
 
  void GP_Vector::Append(double x)
  {
    _data.push_back(x);
  }

  void GP_Vector::Append(GP_Vector const &other)
  {
    uint old_size = _data.size();
    uint new_size = old_size + other._data.size();

    _data.resize(new_size);
    for(uint i=old_size; i<new_size; ++i)
      _data[i] = other._data[i-old_size];
  }

  void GP_Vector::RemoveLast()
  {
    _data.pop_back();
  }

  int GP_Vector::Read(std::string filename, int pos)
  {
    READ_FILE(ifile, filename.c_str());
    ifile.seekg(pos);
    uint size;
    ifile >> size;
    _data.clear();
    _data.resize(size);
    for(uint i=0; i<size; ++i){
      ifile >> _data[i];
    }

    return ifile.tellg();
  }

  void GP_Vector::Write(std::string filename) const
  {
    APPEND_FILE(ofile, filename.c_str());
    ofile << _data.size() << std::endl;
    for(uint i=0; i<_data.size(); ++i)
      ofile << _data[i] << " ";
    ofile << std::endl;
    ofile.close();
  }


  GP_Vector operator* (double x, GP_Vector const &m)
  {
    return m * x;
  }

  GP_Vector operator/ (double x, GP_Vector const &m)
  {
    GP_Vector out(m.Size());

    for(uint i=0; i<out.Size(); ++i)
	out[i] = x / m[i];

    return out;
  }

  std::ostream &operator<<(std::ostream &stream, GP_Vector const &vec)
  {
    uint prec = stream.precision();
    prec = stream.precision(MAX(prec, 4));
    stream.setf(std::ios::fixed, std::ios::floatfield);

    stream << "\t";
    for(uint i=0; i<vec.Size(); i++){
      stream << std::setw(prec + 3) << vec[i] << " ";
    }
    stream << std::flush;
	  
    return stream;
  }

  std::istream &operator>>(std::istream &stream, GP_Vector &vec)
  {
    for(uint i=0; i<vec.Size(); ++i)
      stream >> vec[i];
    return stream;
  }
}
