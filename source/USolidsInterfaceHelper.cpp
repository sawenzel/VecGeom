/*
 * USolidsInterfaceHelper.cpp
 *
 *  Created on: May 20, 2015
 *      Author: swenzel
 */

#ifdef VECGEOM_USOLIDS

#include "base/Global.h"
#include "UVector3.hh"
#include "VUSolid.hh"
#include "volumes/USolidsInterfaceHelper.h"
#undef NDEBUG
#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


double USolidsInterfaceHelper::DistanceToOut(Vector3D<double> const &point,
                               Vector3D<double> const &direction,
                               Vector3D<double> &normal,
                               bool &convex,
                               double stepMax) const {
      assert(0 &&
                 "This DistanceToOut interface was not implemented for this volume.");
      return 0.;
}

std::string USolidsInterfaceHelper::GetEntityType() const {
  assert(0 && "GetEntityType not implemented for USolids interface compatible"
              " volume.");
  return std::string();
}

void USolidsInterfaceHelper::GetParametersList(int number, double *array) const {
  assert(0 && "GetParameterList not implemented for USolids interface"
              " compatible volume.");
}

VUSolid* USolidsInterfaceHelper::Clone() const {
  assert(0 && "Clone not implemented for USolids interface compatible"
              " volume.");
  return NULL;
}

std::ostream& USolidsInterfaceHelper::StreamInfo(std::ostream &os) const {
  assert(0 && "StreamInfo not implemented for USolids interface compatible"
              " volume.");
  return os;
}

UVector3 USolidsInterfaceHelper::GetPointOnSurface() const {
  assert(0 && "GetPointOnSurface not implemented for USolids interface"
              " compatible volume.");
  return UVector3();
}

void USolidsInterfaceHelper::ComputeBBox(UBBox *aBox, bool aStore) {
  assert(0 && "ComputeBBox not implemented for USolids interface compatible"
              " volume.");
}


} // end inline namespace

} // end namespace

#endif
