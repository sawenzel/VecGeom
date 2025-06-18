/*
 * TorusStruct2.h
 *
 *  Created on: 26.06.2017
 *      Author: Aldo Miranda-Aguilar (aldo.nicolas.miranda.aguilar@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_TORUSSTRUCT2_H_
#define VECGEOM_VOLUMES_TORUSSTRUCT2_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/volumes/UnplacedTube.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

using GenericUnplacedTube = SUnplacedTube<TubeTypes::UniversalTube>;
/*
 * A Torus struct without member functions
 *
 */
template <typename T = double>
struct TorusStruct2 {
  // tube defining parameters
  T fRmin;  //< inner radius
  T fRmax;  //< outer radius
  T fRtor;  //< torus radius
  T fSphi;  //< starting phi value (in radians)
  T fDphi;  //< delta phi value of tube segment (in radians)
  T fRmin2; //< inner radius
  T fRmax2; //< outer radius
  T fRtor2; //< torus radius
  evolution::Wedge fPhiWedge;
  GenericUnplacedTube fBoundingTube;

  VECCORE_ATT_HOST_DEVICE
  TorusStruct2(const T rmin, const T rmax, const T rtor, const T sphi, const T dphi)
      : fRmin(rmin), fRmax(rmax), fRtor(rtor), fSphi(sphi), fDphi(dphi), fRmin2(rmin * rmin), fRmax2(rmax * rmax),
        fRtor2(rtor * rtor), fPhiWedge(dphi, sphi),
        fBoundingTube(rtor - rmax - kTolerance, rtor + rmax + kTolerance, rmax, sphi, dphi)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  T rmin() const { return fRmin; }

  VECCORE_ATT_HOST_DEVICE
  T rmax() const { return fRmax; }

  VECCORE_ATT_HOST_DEVICE
  T rtor() const { return fRtor; }

  VECCORE_ATT_HOST_DEVICE
  T sphi() const { return fSphi; }

  VECCORE_ATT_HOST_DEVICE
  T dphi() const { return fDphi; }

  VECCORE_ATT_HOST_DEVICE
  T rmin2() const { return fRmin2; }

  VECCORE_ATT_HOST_DEVICE
  T rmax2() const { return fRmax2; }

  VECCORE_ATT_HOST_DEVICE
  T rtor2() const { return fRtor2; }

  VECCORE_ATT_HOST_DEVICE
  evolution::Wedge GetWedge() const { return fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  GenericUnplacedTube const &GetBoundingTube() const { return fBoundingTube; }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif