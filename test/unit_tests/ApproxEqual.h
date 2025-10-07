// ApproxEqual Functions for geometry test programs
//
// History:
// 20.07.95 P.Kent Translated from old code

#ifndef APPROXEQUAL_HH
#define APPROXEQUAL_HH

#include <cmath>
#include "VecGeom/base/Math.h"

using vecgeom::Precision;

const double kApproxEqualTolerance    = 1E-6;
const double kApproxEqualToleranceFlt = 1E-3;

// Return true if the double x is approximately equal to y
//
// Process:
//
// Return true is x if less than kApproxEqualTolerance from y

// Return true if the 3vector check is approximately equal to target
template <typename Vec_t>
bool ApproxEqual(const Vec_t &check, const Vec_t &target)
{
  return (ApproxEqual(check.x(), target.x()) && ApproxEqual(check.y(), target.y()) &&
          ApproxEqual(check.z(), target.z()))
             ? true
             : false;
}

template <>
bool ApproxEqual<double>(const double &x, const double &y)
{
  if (x == y) {
    return true;
  } else if (x * y == 0.0) {
    double diff = std::fabs(x - y);
    return diff < kApproxEqualTolerance;
  } else if (fabs(x) > 1.0e+100 || fabs(y) > 1.0e+100) {
    // handle comparisons to infinity
    return (x * y > 0) && fabs(x) > 1.0e+100 && fabs(y) > 1.0e+100;
  } else {
    double diff  = std::fabs(x - y);
    double abs_x = std::fabs(x), abs_y = std::fabs(y);
    return diff / (abs_x + abs_y) < kApproxEqualTolerance;
  }
}

template <>
bool ApproxEqual<float>(const float &x, const float &y)
{
  if (x == y) {
    return true;
  } else if (x * y == 0.0) {
    float diff = std::fabs(x - y);
    return diff < kApproxEqualToleranceFlt;
  } else if (fabs(x) > 1.0e+100 || fabs(y) > 1.0e+100) {
    // handle comparisons to infinity
    return (x * y > 0) && fabs(x) > 1.0e+100 && fabs(y) > 1.0e+100;
  } else {
    float diff  = std::fabs(x - y);
    float abs_x = std::fabs(x), abs_y = std::fabs(y);
    return diff / (abs_x + abs_y) < kApproxEqualToleranceFlt;
  }
}

#endif
