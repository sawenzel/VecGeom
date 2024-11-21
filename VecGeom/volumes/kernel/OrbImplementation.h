// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief This file implements the algorithms for Orb
/// \file volumes/kernel/orbImplementation.h
/// \author Raman Sehgal

/// History notes:
/// 2014 - 2015: original development (abstracted kernels); Raman Sehgal
/// July 2016: revision + moving to new backend structure (Raman Sehgal)

#ifndef VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/OrbStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct OrbImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, OrbImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedOrb;
template <typename T>
struct OrbStruct;
class UnplacedOrb;

struct OrbImplementation {

  using PlacedShape_t    = PlacedOrb;
  using UnplacedStruct_t = OrbStruct<Precision>;
  using UnplacedVolume_t = UnplacedOrb;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &orb,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(orb, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &orb,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(orb, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &orb, Vector3D<Real_v> const &localPoint, Bool_v &completelyinside,
      Bool_v &completelyoutside)
  {
    Precision fR = orb.fR;
    Real_v rad2  = localPoint.Mag2();
    Real_v tolR  = fR - Real_v(kTolerance);
    if (ForInside) completelyinside = (rad2 <= tolR * tolR);
    tolR              = fR + Real_v(kTolerance);
    completelyoutside = (rad2 >= tolR * tolR);
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &orb,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    using Bool_v         = vecCore::Mask_v<Real_v>;
    distance             = kInfLength;
    Real_v rad           = point.Mag();
    Bool_v isPointInside = (rad < Real_v(orb.fR - kTolerance));
    vecCore__MaskedAssignFunc(distance, isPointInside, Real_v(-1.));
    Bool_v done = isPointInside;
    if (vecCore::MaskFull(done)) return;

    Real_v pDotV3D          = point.Dot(direction);
    Bool_v isPointOnSurface = (rad >= Real_v(orb.fR - kTolerance)) && (rad <= Real_v(orb.fR + kTolerance));
    Bool_v cond             = (isPointOnSurface && (pDotV3D < Real_v(0.)));
    vecCore__MaskedAssignFunc(distance, !done && cond, Real_v(0.));
    done |= cond;
    if (vecCore::MaskFull(done)) return;
    Real_v dist(kInfLength);
    vecCore::MaskedAssign(
        distance, !done && DetectIntersectionAndCalculateDistance<Real_v, true>(orb, point, direction, dist), dist);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &orb,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;

    distance = kInfLength;

    Real_v rad            = point.Mag();
    Bool_v isPointOutside = (rad > Real_v(orb.fR + kTolerance));
    vecCore__MaskedAssignFunc(distance, isPointOutside, Real_v(-1.));
    Bool_v done = isPointOutside;
    if (vecCore::MaskFull(done)) return;

    Real_v pDotV3D          = point.Dot(direction);
    Bool_v isPointOnSurface = (rad >= Real_v(orb.fR - kTolerance)) && (rad <= Real_v(orb.fR + kTolerance));
    Bool_v cond             = (isPointOnSurface && (pDotV3D > Real_v(0.)));
    vecCore__MaskedAssignFunc(distance, !done && cond, Real_v(0.));
    done |= cond;
    if (vecCore::MaskFull(done)) return;
    Real_v dist(kInfLength);
    vecCore::MaskedAssign(
        distance, !done && DetectIntersectionAndCalculateDistance<Real_v, false>(orb, point, direction, dist), dist);

    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &orb,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    using Bool_v         = vecCore::Mask_v<Real_v>;
    Real_v rad           = point.Mag();
    safety               = rad - Real_v(orb.fR);
    Bool_v isPointInside = (rad < Real_v(orb.fR - kTolerance));
    vecCore__MaskedAssignFunc(safety, isPointInside, Real_v(-1.));
    if (vecCore::MaskFull(isPointInside)) return;

    Bool_v isPointOnSurface = (rad > Real_v(orb.fR - kTolerance)) && (rad < Real_v(orb.fR + kTolerance));
    vecCore__MaskedAssignFunc(safety, isPointOnSurface, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &orb,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v rad = point.Mag();
    safety     = Real_v(orb.fR) - rad;

    Bool_v isPointOutside = (rad > Real_v(orb.fR + kTolerance));
    vecCore__MaskedAssignFunc(safety, isPointOutside, Real_v(-1.));
    if (vecCore::MaskFull(isPointOutside)) return;

    Bool_v isPointOnSurface = (rad > Real_v(orb.fR - kTolerance)) && (rad < Real_v(orb.fR + kTolerance));
    vecCore__MaskedAssignFunc(safety, isPointOnSurface, Real_v(0.));
  }

  template <typename Real_v, bool ForDistanceToIn>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static typename vecCore::Mask_v<Real_v>
  DetectIntersectionAndCalculateDistance(UnplacedStruct_t const &orb, Vector3D<Real_v> const &point,
                                         Vector3D<Real_v> const &direction, Real_v &distance)
  {

    using Bool_v   = vecCore::Mask_v<Real_v>;
    Real_v rad2    = point.Mag2();
    Real_v pDotV3D = point.Dot(direction);
    Precision fR   = orb.fR;
    Real_v c       = rad2 - fR * fR;
    Real_v d2      = (pDotV3D * pDotV3D - c);

    if (ForDistanceToIn) {
      Bool_v cond = ((d2 >= Real_v(0.)) && (pDotV3D <= Real_v(0.)));
      vecCore__MaskedAssignFunc(distance, cond, (-pDotV3D - Sqrt(vecCore::math::Abs(d2))));
      return cond;
    } else {
      vecCore__MaskedAssignFunc(distance, (d2 >= Real_v(0.)), (-pDotV3D + Sqrt(vecCore::math::Abs(d2))));
      return (d2 >= Real_v(0.));
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &orb, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    Real_v rad2             = point.Mag2();
    Real_v invRadius        = Real_v(1.) / Sqrt(rad2);
    Vector3D<Real_v> normal = point * invRadius;

    Real_v tolRMaxO = orb.fR + kTolerance;
    Real_v tolRMaxI = orb.fR - kTolerance;

    // Check radial surface
    valid = ((rad2 <= tolRMaxO * tolRMaxO) && (rad2 >= tolRMaxI * tolRMaxI)); // means we are on surface
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_orbIMPLEMENTATION_H_
