#include "GPlib/GP_OptimizerCG.hh"
#include <limits>

namespace GPLIB {

  GP_OptimizerCG::GP_OptimizerCG(GP_ObjectiveFunctionBase &fn, 
				 int length, double eps) :
    _interv(0.1), _ext(3.0), _ratio(10.), _sig(0.1), _rho(_sig/2.), 
    _conv_thresh(eps), _max(20),
    _objFn(fn), _iter(0), _length(length), _method(length > 0 ? "linesearch" : "function eval")
  {
  }
  
  void GP_OptimizerCG::Init(std::vector<double> const &init)
  {
    _n = _objFn.GetNbArgs();
    _ls_failed = false;
    _red = 1.;

    std::pair<double, GP_Vector> fdf = _objFn.ValAndDeriv(init);
    _f0 = fdf.first;
    _df0 = fdf.second;

    _X = _Z = GP_Vector(init);
    _fX = std::vector<double>(1, _f0);
    if(_length < 0) ++_iter;
    _s = -_df0;
    _d0 = -_s.Dot(_s);
    _x3 = _red / (1. - _d0);
  }


  bool GP_OptimizerCG::Iterate()
  {
    if(_length > 0) ++_iter;    
    
    GP_Vector X0 = _X, dF0 = _df0;
    double F0 = _f0;
    
    uint M;
    if(_length > 0) 
      M = _max;
    else 
      M = MIN(_max, -_length - _iter);

    Extrapolation(M, X0, dF0, F0);
    Interpolation(M, X0, dF0, F0);
    if(!NewSearchDirection(M, X0, dF0, F0))
      return false;

    return _iter < fabs(_length);
  }
  
  void GP_OptimizerCG::Extrapolation(uint &M, GP_Vector &X0, GP_Vector &dF0,
				     double &F0)
  {
    while(1)  {
      // keep extrapolating as long as necessary
      _x2 = 0; _f2 = _f0; _d2 = _d0; _f3 = _f0; 
      _df3 = _df0;

      bool success = false;
      while (!success && M > 0){
	try {
	  --M; _iter += (_length<0);  // count epochs?!

	  std::pair<double, GP_Vector> fdf = _objFn.ValAndDeriv(_X + _x3 * _s);
	  _f3 = fdf.first;
	  _df3 = fdf.second;
        
	  if (std::isnan(_f3) || std::isinf(_f3) || 
	      _df3.AnyIsNaN() || _df3.AnyIsInf()){
	    throw GP_EXCEPTION("Error: NaN or Inf");
	  }
	  success = true;
	}
	// catch any error which occured in f
	catch(GP_Exception e) {
	  // bisect and try again
	  _x3 = (_x2+_x3) / 2; 
        }
      }

      if (_f3 < F0){
	// keep best values
	X0 = _X + _x3 * _s; 
	F0 = _f3; 
	dF0 = _df3; 
      }
      // new slope
      _d3 = _df3.Dot(_s);

      if (_d3 > _sig*_d0 || _f3 > _f0+_x3*_rho*_d0 || 
	  M == 0) { // are we done extrapolating?
	return;
      }
      
      // move point 2 to point 1
      _x1 = _x2; _f1 = _f2; _d1 = _d2; 
      // move point 3 to point 2
      _x2 = _x3; _f2 = _f3; _d2 = _d3;
      // make cubic extrapolation
      double A = 6*(_f1 - _f2) + 3*(_d2 + _d1)*(_x2 - _x1);
      double B = 3*(_f2 - _f1) - (2*_d1 + _d2)*(_x2 - _x1);
      // num. error possible, ok!
      _x3 = _x1 - _d1*SQR(_x2 - _x1) / (B + sqrt(B*B - A*_d1*(_x2 - _x1))); 
      // num prob | wrong sign?
      if (std::isnan(_x3) || std::isinf(_x3) || _x3 < 0) 
	_x3 = _x2*_ext;                       // extrapolate maximum amount
      else if (_x3 > _x2*_ext)                // new point beyond extrapolation limit?
	_x3 = _x2*_ext;                       // extrapolate maximum amount
      else if (_x3 < _x2+_interv*(_x2 - _x1))     // new point too close to previous point?
	_x3 = _x2 + _interv*(_x2 - _x1);
    }                                        // end extrapolation
  }

  void GP_OptimizerCG::Interpolation(uint &M, GP_Vector &X0, GP_Vector &dF0,
				     double &F0)
  {
    while ((fabs(_d3) > -_sig*_d0 || 
	    _f3 > _f0+_x3*_rho*_d0) && M > 0){                  // keep interpolating
      if (_d3 > 0 || _f3 > _f0 + _x3*_rho*_d0){                 // choose subinterval
	_x4 = _x3; _f4 = _f3; _d4 = _d3;                   // move point 3 to point 4
      }          
      else {
	_x2 = _x3; _f2 = _f3; _d2 = _d3;                  // move point 3 to point 2
      }
      if (_f4 > _f0)           
	_x3 = _x2-(0.5*_d2*SQR(_x4-_x2))/(_f4-_f2-_d2*(_x4-_x2));  // quadratic interpolation
      else {
	double A = 6*(_f2 - _f4)/(_x4 - _x2) + 3*(_d4 + _d2);                    // cubic interpolation
	double B = 3*(_f4 - _f2)-(2*_d2 + _d4)*(_x4 - _x2);
	_x3 = _x2 + (sqrt(B*B-A*_d2*SQR(_x4 -_x2))-B)/A;        // num. error possible, ok!
      }
      if (std::isnan(_x3) || std::isinf(_x3))
	_x3 = (_x2+_x4)/2;               // if we had a numerical problem then bisect

      _x3 = MAX(MIN(_x3, _x4-_interv*(_x4-_x2)), _x2+_interv*(_x4-_x2));  // don't accept too close

      std::pair<double, GP_Vector> fdf = _objFn.ValAndDeriv(_X + _x3 * _s);
      _f3 = fdf.first;
      _df3 = fdf.second;

      if (_f3 < F0){
	// keep best values
	X0 = _X + _x3*_s; 
	F0 = _f3; 
	dF0 = _df3; 
      }        
      --M; 
      _iter += (_length<0);                             // count epochs?!
      _d3 = _df3.Dot(_s);                                                    // new slope
    }             
  }

  bool GP_OptimizerCG::NewSearchDirection(uint &M, GP_Vector const &X0, 
					  GP_Vector const &dF0, double F0)
  {
    if (fabs(_d3) < -_sig*_d0 && _f3 < _f0 + _x3*_rho*_d0) {         // if line search succeeded
      _X += _x3*_s; 
      _f0 = _f3; 
      _fX.push_back(_f0);
      
      _s = (_df3.Dot(_df3) - _df0.Dot(_df3))/(_df0.Dot(_df0))*_s - _df3;   // Polack-Ribiere CG direction
      _df0 = _df3;                                               // swap derivatives
      _d3 = _d0; _d0 = _df0.Dot(_s);
      if (_d0 > 0) {                                    // new slope must be negative
	_s = -_df0; _d0 = -_s.Dot(_s);                  // otherwise use steepest direction
      }
      _x3 = _x3 * MIN(_ratio, _d3/(_d0-std::numeric_limits<double>::min()));          // slope ratio but max RATIO
      _ls_failed = false;                              // this line search did not fail
    }
    else {
      _X = X0; _f0 = F0; _df0 = dF0;                     // restore best point so far
      if (_ls_failed || _iter > abs(_length))         // line search failed twice in a row
	return false;                             // or we ran out of time, so we give up
    
      _s = -_df0; _d0 = -_s.Dot(_s);                                        // try steepest
      _x3 = 1/(1 - _d0);                     
      _ls_failed = true;                                    // this line search failed
    }
    return true;
  }

  void GP_OptimizerCG::GetCurrValues(std::vector<double> &vals) const
  {
    vals = _X;
  }
  
  double GP_OptimizerCG::GetCurrentError() const
  {
    return _f0;
  }
  
  bool  GP_OptimizerCG::TestConvergence() const
  {
    return (_df0.Norm() < _conv_thresh);
  }
  
}
