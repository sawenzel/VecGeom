/*
 * ConeStruct.h
 *
 *  Created on: May 11, 2017
 *      Author: Raman Sehgal
 */
#ifndef VECGEOM_POLYCONESECTION_H_
#define VECGEOM_POLYCONESECTION_H_

#include "VecGeom/volumes/ConeStruct.h"

namespace vecgeom {

// helper structure to encapsulate a section
VECGEOM_DEVICE_FORWARD_DECLARE(struct PolyconeSection;);
VECGEOM_DEVICE_DECLARE_CONV(struct, PolyconeSection);

inline namespace VECGEOM_IMPL_NAMESPACE {

struct PolyconeSection {
  VECCORE_ATT_HOST_DEVICE
  PolyconeSection() : fShift(0.0), fTubular(0), fConvex(0) {}

  VECCORE_ATT_HOST_DEVICE
  ~PolyconeSection() {}

  ConeStruct<Precision> fSolid;
  Precision fShift;
  bool fTubular;
  bool fConvex; // TRUE if all points in section are concave in regards to whole polycone, will be determined
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
