// ApproxEqual Functions for geometry test programs
//
// History:
// 20.07.95 P.Kent Translated from old code

#ifndef APPROXEQUAL_HH
#define APPROXEQUAL_HH

const double kApproxEqualTolerance = 1E-6;

// Return true if the double x is approximately equal to y
//
// Process:
//
// Return true is x if less than kApproxEqualTolerance from y

bool ApproxEqual(const double x, const double y)
{
	if (x == y) {
		return true;
	} else if (x * y == 0.0) {
		double diff = std::fabs(x - y);
		return diff < kApproxEqualTolerance;
	} else {
		double diff = std::fabs(x - y);
		double abs_x = std::fabs(x), abs_y = std::fabs(y);
		return diff / (abs_x + abs_y) < kApproxEqualTolerance;
	}
}

// Return true if the 3vector check is approximately equal to target
template <class Vec_t>
bool ApproxEqual(const Vec_t& check, const Vec_t& target)
{
  return (ApproxEqual(check.x(),target.x())&&
	  ApproxEqual(check.y(),target.y())&&
	  ApproxEqual(check.z(),target.z())) ? true : false;
}


#endif









