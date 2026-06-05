//===-- kernel/HypeImplementation.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// @file HypeImplementation.h
/// @brief Navigation kernels for the hyperboloid shape.
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/HypeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>
#include "VecGeom/volumes/HypeUtilities.h"
#include "VecGeom/volumes/kernel/shapetypes/HypeTypes.h"

// different SafetyToIn implementations
// #define ACCURATE_BB
#define ACCURATE_BC

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, HypeImplementation, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
struct HypeStruct;

template <typename T>
class SPlacedHype;
template <typename T>
class SUnplacedHype;

/// @brief Implements Hype navigation and classification kernels.
/// @details The template parameter selects whether the inner hyperbolic surface
/// is known to exist at compile time or must be checked from the runtime
/// `HypeStruct`.
template <typename hypeTypeT>
struct HypeImplementation {

  using UnplacedStruct_t = HypeStruct<Precision>;
  using UnplacedVolume_t = SUnplacedHype<hypeTypeT>;
  using PlacedShape_t    = SPlacedHype<UnplacedVolume_t>;

  /// @brief Test whether a local point is contained in or on the Hype.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Bool_t Boolean output type.
  /// @param hype Hype runtime data.
  /// @param point Local point to test.
  /// @param[out] inside Set to true unless the point is completely outside.
  template <typename Real_v, typename Bool_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &hype,
                                                                    Vector3D<Real_v> const &point, Bool_t &inside)
  {
    bool unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, false>(hype, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Inside_t Integer-like type used for `EInside` values.
  /// @param hype Hype runtime data.
  /// @param point Local point to classify.
  /// @param[out] inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &hype,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    inside = EInside::kSurface;
    bool completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, true>(hype, point, completelyinside, completelyoutside);
    if (completelyoutside) inside = EInside::kOutside;
    if (completelyinside) inside = EInside::kInside;
  }

  /// @brief Compute strict inside/outside flags for point classification.
  /// @details Classification combines z extent, outer hyperbolic surface, and
  /// the optional inner hyperbolic surface. Points in tolerance bands can leave
  /// both flags false.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam ForInside When true, also compute the strict-inside flag.
  /// @param hype Hype runtime data.
  /// @param point Local point to classify.
  /// @param[out] completelyinside Set when the point is strictly inside.
  /// @param[out] completelyoutside Set when the point is clearly outside.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &hype, Vector3D<Real_v> const &point, bool &completelyinside, bool &completelyoutside)
  {
    using namespace ::vecgeom::HypeTypes;
    Real_v r2    = point.Perp2();
    Real_v oRad2 = (hype.fRmax2 + hype.fTOut2 * point.z() * point.z());
    Real_v iRad2(0.);

    completelyoutside = (Abs(point.z()) > (hype.fDz + hype.zToleranceLevel));
    if (completelyoutside) return;
    completelyoutside |= (r2 > oRad2 + hype.outerRadToleranceLevel);
    if (completelyoutside) return;
    if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) {
      iRad2 = (hype.fRmin2 + hype.fTIn2 * point.z() * point.z());
      completelyoutside |= (r2 < (iRad2 - hype.innerRadToleranceLevel));
    }
    if (completelyoutside) return;

    if (ForInside) {
      completelyinside =
          (Abs(point.z()) < (hype.fDz - hype.zToleranceLevel)) && (r2 < oRad2 - hype.outerRadToleranceLevel);

      if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) completelyinside &= (r2 > (iRad2 + hype.innerRadToleranceLevel));
    }
  }

  /// @brief Compute the first valid entry distance.
  /// @details Surface starts moving into material return zero, clearly inside
  /// starts return `-1`, and misses keep `kInfLength`. Candidate hits are
  /// checked against the z caps and active inner/outer hyperbolic surfaces.
  /// @tparam Real_v Floating-point scalar type.
  /// @param hype Hype runtime data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Entry distance, `0`, `-1`, or `kInfLength`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &hype,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    using namespace ::vecgeom::HypeTypes;
    Real_v absZ = Abs(point.z());
    distance    = kInfLength;
    Real_v zDist(kInfLength), dist(kInfLength);
    Real_v r = point.Perp2();

    bool surfaceCond = HypeUtilities::IsPointOnSurfaceAndMovingInside<Real_v, hypeTypeT>(hype, point, direction);
    if (surfaceCond) {
      distance = Real_v(0.0);
      return;
    }

    bool cond = HypeUtilities::IsCompletelyInside<Real_v, hypeTypeT>(hype, point);
    if (cond) {
      distance = Real_v(-1.0);
      return;
    }

    bool isPointAboveOrBelowHypeAndGoingInside = (absZ > hype.fDz) && (point.z() * direction.z() < Real_v(0.));
    Real_v rp2(0.);
    if (isPointAboveOrBelowHypeAndGoingInside) {
      bool hittingZPlane =
          HypeUtilities::GetPointOfIntersectionWithZPlane<Real_v, hypeTypeT, true>(hype, point, direction, zDist);
      if (hittingZPlane) {
        distance = zDist;
        return;
      }
      Real_v x = point.x() + zDist * direction.x();
      Real_v y = point.y() + zDist * direction.y();
      rp2      = x * x + y * y;
    }

    bool hittingOuterSurfaceFromOutsideZRange = isPointAboveOrBelowHypeAndGoingInside && (rp2 >= hype.fEndOuterRadius2);
    bool hittingOuterSurfaceFromWithinZRange  = ((r > ((hype.fRmax2 + hype.fTOut2 * absZ * absZ) + kHalfTolerance))) &&
                                                (absZ >= Real_v(0.)) && (absZ <= hype.fDz);

    cond = (hittingOuterSurfaceFromOutsideZRange || hittingOuterSurfaceFromWithinZRange ||
            (HypeUtilities::IsPointOnOuterSurfaceAndMovingOutside<Real_v>(hype, point, direction))) &&
           HypeHelpers<Real_v, true, false>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
    if (cond) distance = dist;

    if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) {
      if (cond) return;
      bool hittingInnerSurfaceFromOutsideZRange =
          isPointAboveOrBelowHypeAndGoingInside && (rp2 <= hype.fEndInnerRadius2);
      bool hittingInnerSurfaceFromWithinZRange = (r < ((hype.fRmin2 + hype.fTIn2 * absZ * absZ) - kHalfTolerance)) &&
                                                 (absZ >= Real_v(0.)) && (absZ <= hype.fDz);

      // If it hits inner hyperbolic surface then distance will be the distance to inner hyperbolic surface
      // Or if the point is on the inner Hyperbolic surface but going out then the distance will be the distance to
      // opposite inner hyperbolic surface.
      cond = (hittingInnerSurfaceFromOutsideZRange || hittingInnerSurfaceFromWithinZRange ||
              (HypeUtilities::IsPointOnInnerSurfaceAndMovingOutside<Real_v, hypeTypeT>(hype, point, direction))) &&
             HypeHelpers<Real_v, true, true>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
      if (cond) distance = dist;
    }
  }

  /// @brief Compute the first valid exit distance.
  /// @details Surface starts moving out of material return zero, clearly
  /// outside starts return `-1`, and misses keep `kInfLength`. Exit candidates
  /// are the z cap, outer hyperbolic surface, and optional inner surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @param hype Hype runtime data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Exit distance, `0`, `-1`, or `kInfLength`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &hype,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    using namespace ::vecgeom::HypeTypes;
    distance = kInfLength;
    Real_v zDist(kInfLength), dist(kInfLength);

    bool cond = HypeUtilities::IsPointOnSurfaceAndMovingOutside<Real_v, hypeTypeT>(hype, point, direction);
    if (cond) {
      distance = Real_v(0.);
      return;
    }

    cond = HypeUtilities::IsCompletelyOutside<Real_v, hypeTypeT>(hype, point);
    if (cond) {
      distance = Real_v(-1.);
      return;
    }

    HypeUtilities::GetPointOfIntersectionWithZPlane<Real_v, hypeTypeT, false>(hype, point, direction, zDist);
    if (zDist < Real_v(0.)) zDist = InfinityLength<Real_v>();

    HypeHelpers<Real_v, false, false>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
    if (dist < Real_v(0.)) dist = InfinityLength<Real_v>();
    distance = Min(zDist, dist);

    if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) {
      HypeHelpers<Real_v, false, true>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
      if (dist < Real_v(0.)) dist = InfinityLength<Real_v>();
      distance = Min(distance, dist);
    }
  }

  /// @brief Compute safety from an exterior point to the Hype.
  /// @details Returns `-1` for clearly inside starts, zero for points in the
  /// classification tolerance region, and otherwise estimates the nearest cap
  /// or hyperbolic-surface approach.
  /// @tparam Real_v Floating-point scalar type.
  /// @param hype Hype runtime data.
  /// @param point Local point, expected outside.
  /// @param[out] safety Safety estimate to the solid.
  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point,
                                                 Real_v &safety)
  {
    Real_v absZ = Abs(point.z());
    Real_v r2   = point.Perp2();
    Real_v r    = Sqrt(r2);

    // New Simple Algo
    safety = 0.;
    // If point is inside then safety should be -1.
    bool compIn(false), compOut(false);
    GenericKernelForContainsAndInside<Real_v, true>(hype, point, compIn, compOut);
    if (!compIn && !compOut) return;

    if (compIn) {
      safety = Real_v(-1.0);
      return;
    }

    Real_v sigz = absZ - hype.fDz;
    if (sigz > kHalfTolerance) {
      if (r < hype.fEndOuterRadius && r > hype.fEndInnerRadius) {
        safety = sigz;
        return;
      }
      if (r > hype.fEndOuterRadius) {
        Real_v dr = r - hype.fEndOuterRadius;
        safety    = Sqrt(dr * dr + sigz * sigz);
        return;
      }
      if (r < hype.fEndInnerRadius) {
        Real_v dr = r - hype.fEndInnerRadius;
        safety    = Sqrt(dr * dr + sigz * sigz);
        return;
      }
    }

    if (absZ > Real_v(0.) && absZ < hype.fDz) {
      Real_v outerRad2 = hype.fRmax2 + hype.fTOut2 * absZ * absZ;
      if (r2 > outerRad2 + kHalfTolerance) {
        safety = HypeUtilities::ApproxDistOutside<Real_v>(r, absZ, hype.fRmax, hype.fTOut);
        return;
      }

      Real_v innerRad2 = hype.fRmin2 + hype.fTIn2 * absZ * absZ;
      if (r2 < innerRad2 - kHalfTolerance) {
        safety = HypeUtilities::ApproxDistInside<Real_v>(r, absZ, hype.fRmin, hype.fTIn2);
      }
    }
  }

  /// @brief Compute safety from an interior point to leave the Hype.
  /// @details Returns `-1` for clearly outside starts, zero for points in the
  /// classification tolerance region, and otherwise the minimum of cap, outer
  /// surface, and optional inner-surface safety estimates.
  /// @tparam Real_v Floating-point scalar type.
  /// @param hype Hype runtime data.
  /// @param point Local point, expected inside.
  /// @param[out] safety Safety estimate to the boundary.
  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point,
                                                  Real_v &safety)
  {
    using namespace ::vecgeom::HypeTypes;
    safety      = Real_v(0.);
    Real_v r    = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v absZ = Abs(point.z());
    bool inside(false), outside(false);

    GenericKernelForContainsAndInside<Real_v, true>(hype, point, inside, outside);
    if (!inside && !outside) return;

    if (outside) {
      safety = Real_v(-1.0);
      return;
    }

    Real_v distZ               = Abs(absZ - hype.fDz);
    Real_v distInner           = Real_v(0.);
    const bool hasInnerSurface = checkInnerSurfaceTreatment<hypeTypeT>(hype);
    if (hasInnerSurface) {
      distInner =
          hype.fStIn ? HypeUtilities::ApproxDistOutside<Real_v>(r, absZ, hype.fRmin, hype.fTIn) : (r - hype.fRmin);
    } else if (!hype.fStIn) {
      distInner = InfinityLength<Real_v>();
    }

    Real_v distOuter = HypeUtilities::ApproxDistInside<Real_v>(r, absZ, hype.fRmax, hype.fTOut2);
    safety           = Min(distInner, distOuter);
    safety           = Min(safety, distZ);
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
