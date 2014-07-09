// ApproxEqual Functions for geometry test programs
//
// History:
// 20.07.95 P.Kent Translated from old code

#ifndef APPROXEQUAL_HH
#define APPROXEQUAL_HH

#include "UVector3.hh"

const double kApproxEqualTolerance = 1E-6;

// Return true if the double check is approximately equal to target
//
// Process:
//
// Return true is check if less than kApproxEqualTolerance from target

bool ApproxEqual(const double check,const double target)
{
    return (std::fabs(check-target)<kApproxEqualTolerance) ?true:false;
}

// Return true if the 3vector check is approximately equal to target
bool ApproxEqual(const UVector3& check, const UVector3& target)
{
  return (ApproxEqual(check.x(),target.x())&&
	  ApproxEqual(check.y(),target.y())&&
	  ApproxEqual(check.z(),target.z())) ? true : false;
}


#endif









