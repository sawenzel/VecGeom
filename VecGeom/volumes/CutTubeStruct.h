/*
 * CutTubeStruct.h
 *
 *  Created on: 03.11.2016
 *      Author: mgheata
 */
#ifndef VECGEOM_CUTTUBESTRUCT_H_
#define VECGEOM_CUTTUBESTRUCT_H_

#include <VecCore/VecCore>
#include "VecGeom/base/Vector3D.h"

#include "TubeStruct.h"
#include "CutPlanes.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Cached geometric state for a cut tube.
 *
 * The embedded tube helper uses the `kInfLength` z sentinel because the cut
 * planes, not flat z planes, own the longitudinal boundaries.
 */
template <typename T = double>
struct CutTubeStruct {
  T fDz;                     ///< Z half length
  TubeStruct<T> fTubeStruct; ///< Radial / phi helper state with infinite-z sentinel
  CutPlanes fCutPlanes;      ///< Top and bottom cut planes

  T fCosPhi1; //< Cosine of phi
  T fSinPhi1; //< Sine of phi
  T fCosPhi2; //< Cosine of phi+dphi
  T fSinPhi2; //< Sine of phi+dphi
  T fMaxVal;

  VECCORE_ATT_HOST_DEVICE
  CutTubeStruct() : fTubeStruct(0., 0., 0., 0., 0.), fCutPlanes() {}

  VECCORE_ATT_HOST_DEVICE
  CutTubeStruct(T const &rmin, T const &rmax, T const &z, T const &sphi, T const &dphi, Vector3D<T> const &bottomNormal,
                Vector3D<T> const &topNormal)
      : fDz(z), fTubeStruct(rmin, rmax, kInfLength, sphi, dphi), fCutPlanes()
  {
    fCutPlanes.Set(0, bottomNormal.Unit(), Vector3D<T>(0., 0., -z));
    fCutPlanes.Set(1, topNormal.Unit(), Vector3D<T>(0., 0., z));
    fCosPhi1 = vecCore::math::Cos(sphi);
    fSinPhi1 = vecCore::math::Sin(sphi);
    fCosPhi2 = vecCore::math::Cos(sphi + dphi);
    fSinPhi2 = vecCore::math::Sin(sphi + dphi);
    fMaxVal  = vecCore::math::Max(rmax, z);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TubeStruct<T> const &GetTubeStruct() const { return fTubeStruct; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  CutPlanes const &GetCutPlanes() const { return fCutPlanes; }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
