#include "GPlib/GP_Exception.hh"
#include <iostream>

namespace GPLIB {

  void GP_Exception::Handle() const
  {
    std::cerr << "Error! In " << _filename << " line " << _line_number << ": " << _message << std::endl;
  }


}

