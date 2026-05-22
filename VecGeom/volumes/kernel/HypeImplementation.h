//===-- kernel/HypeImplementation.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file implements the Hype shape
///

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

template <typename hypeTypeT>
struct HypeImplementation {

  using UnplacedStruct_t = HypeStruct<Precision>;
  using UnplacedVolume_t = SUnplacedHype<hypeTypeT>;
  using PlacedShape_t    = SPlacedHype<UnplacedVolume_t>;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &hype,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(hype, point, unused, outside);
    inside = !outside;
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &hype,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_v>;
    Bool_v completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(hype, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_v(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_v(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &hype, Vector3D<Real_v> const &point, Bool_v &completelyinside, Bool_v &completelyoutside)
  {
    using namespace ::vecgeom::HypeTypes;
    Real_v r2    = point.Perp2();
    Real_v oRad2 = (hype.fRmax2 + hype.fTOut2 * point.z() * point.z());
    Real_v iRad2(0.);

    completelyoutside = (Abs(point.z()) > (hype.fDz + hype.zToleranceLevel));
    if (vecCore::MaskFull(completelyoutside)) return;
    completelyoutside |= (r2 > oRad2 + hype.outerRadToleranceLevel);
    if (vecCore::MaskFull(completelyoutside)) return;
    if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) {
      iRad2 = (hype.fRmin2 + hype.fTIn2 * point.z() * point.z());
      completelyoutside |= (r2 < (iRad2 - hype.innerRadToleranceLevel));
    }
    if (vecCore::MaskFull(completelyoutside)) return;

    if (ForInside) {
      completelyinside =
          (Abs(point.z()) < (hype.fDz - hype.zToleranceLevel)) && (r2 < oRad2 - hype.outerRadToleranceLevel);

      if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) completelyinside &= (r2 > (iRad2 + hype.innerRadToleranceLevel));
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &hype,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    using namespace ::vecgeom::HypeTypes;
    using Bool_v = vecCore::Mask_v<Real_v>;
    Real_v absZ  = Abs(point.z());
    distance     = kInfLength;
    Real_v zDist(kInfLength), dist(kInfLength);
    Real_v r = point.Perp2();

    Bool_v done(false);
    Bool_v cond(false);

    Bool_v surfaceCond = HypeUtilities::IsPointOnSurfaceAndMovingInside<Real_v, hypeTypeT>(hype, point, direction);
    vecCore__MaskedAssignFunc(distance, !done && surfaceCond, Real_v(0.0));
    done |= surfaceCond;
    if (vecCore::MaskFull(done)) return;

    cond = HypeUtilities::IsCompletelyInside<Real_v, hypeTypeT>(hype, point);
    vecCore__MaskedAssignFunc(distance, !done && cond, Real_v(-1.0));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    // checking whether point hits the Z Surface of hyperboloid
    Bool_v hittingZPlane =
        HypeUtilities::GetPointOfIntersectionWithZPlane<Real_v, hypeTypeT, true>(hype, point, direction, zDist);
    Bool_v isPointAboveOrBelowHypeAndGoingInside = (absZ > hype.fDz) && (point.z() * direction.z() < Real_v(0.));
    cond                                         = isPointAboveOrBelowHypeAndGoingInside && hittingZPlane;
    vecCore::MaskedAssign(distance, !done && cond, zDist);
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    // Moving the point to Z Surface
    Vector3D<Real_v> newPt = point + zDist * direction;
    Real_v rp2             = newPt.Perp2();

    Bool_v hittingOuterSurfaceFromOutsideZRange =
        isPointAboveOrBelowHypeAndGoingInside && !hittingZPlane && (rp2 >= hype.fEndOuterRadius2);
    Bool_v hittingOuterSurfaceFromWithinZRange = ((r > ((hype.fRmax2 + hype.fTOut2 * absZ * absZ) + kHalfTolerance))) &&
                                                 (absZ >= Real_v(0.)) && (absZ <= hype.fDz);

    cond = (hittingOuterSurfaceFromOutsideZRange || hittingOuterSurfaceFromWithinZRange ||
            (HypeUtilities::IsPointOnOuterSurfaceAndMovingOutside<Real_v>(hype, point, direction))) &&
           HypeHelpers<Real_v, true, false>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
    vecCore::MaskedAssign(distance, !done && cond, dist);

    if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) {
      done |= cond;
      if (vecCore::MaskFull(done)) return;
      Bool_v hittingInnerSurfaceFromOutsideZRange =
          isPointAboveOrBelowHypeAndGoingInside && !hittingZPlane && (rp2 <= hype.fEndInnerRadius2);
      Bool_v hittingInnerSurfaceFromWithinZRange = (r < ((hype.fRmin2 + hype.fTIn2 * absZ * absZ) - kHalfTolerance)) &&
                                                   (absZ >= Real_v(0.)) && (absZ <= hype.fDz);

      // If it hits inner hyperbolic surface then distance will be the distance to inner hyperbolic surface
      // Or if the point is on the inner Hyperbolic surface but going out then the distance will be the distance to
      // opposite inner hyperbolic surface.
      cond = (hittingInnerSurfaceFromOutsideZRange || hittingInnerSurfaceFromWithinZRange ||
              (HypeUtilities::IsPointOnInnerSurfaceAndMovingOutside<Real_v, hypeTypeT>(hype, point, direction))) &&
             HypeHelpers<Real_v, true, true>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
      vecCore::MaskedAssign(distance, !done && cond, dist);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &hype,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    using namespace ::vecgeom::HypeTypes;
    using Bool_v = typename vecCore::Mask_v<Real_v>;
    distance     = kInfLength;
    Real_v zDist(kInfLength), dist(kInfLength);

    Bool_v done(false);

    Bool_v cond = HypeUtilities::IsPointOnSurfaceAndMovingOutside<Real_v, hypeTypeT>(hype, point, direction);
    vecCore__MaskedAssignFunc(distance, cond, Real_v(0.));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    cond = HypeUtilities::IsCompletelyOutside<Real_v, hypeTypeT>(hype, point);
    vecCore__MaskedAssignFunc(distance, !done && cond, Real_v(-1.));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    HypeUtilities::GetPointOfIntersectionWithZPlane<Real_v, hypeTypeT, false>(hype, point, direction, zDist);
    vecCore__MaskedAssignFunc(zDist, zDist < Real_v(0.), InfinityLength<Real_v>());

    HypeHelpers<Real_v, false, false>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
    vecCore__MaskedAssignFunc(dist, dist < Real_v(0.), InfinityLength<Real_v>());
    vecCore__MaskedAssignFunc(distance, !done, Min(zDist, dist));

    if (checkInnerSurfaceTreatment<hypeTypeT>(hype)) {
      HypeHelpers<Real_v, false, true>::GetPointOfIntersectionWithHyperbolicSurface(hype, point, direction, dist);
      vecCore__MaskedAssignFunc(dist, dist < Real_v(0.), InfinityLength<Real_v>());
      vecCore__MaskedAssignFunc(distance, !done, Min(distance, dist));
    }
  }

  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point,
                                                 Real_v &safety)
  {

    using Bool_v = typename vecCore::Mask_v<Real_v>;

    Real_v absZ = Abs(point.z());
    Real_v r2   = point.Perp2();
    Real_v r    = Sqrt(r2);

    Bool_v done(false);

    // New Simple Algo
    safety = 0.;
    // If point is inside then safety should be -1.
    Bool_v compIn(false), compOut(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(hype, point, compIn, compOut);
    done = (!compIn && !compOut);
    if (vecCore::MaskFull(done)) return;

    vecCore__MaskedAssignFunc(safety, compIn, Real_v(-1.0));
    done |= compIn;
    if (vecCore::MaskFull(done)) return;

    Bool_v cond(false);
    Real_v sigz = absZ - hype.fDz;
    if constexpr (vecCore::VectorSize<Real_v>() == 1) {
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
      return;
    }

    cond = (sigz > kHalfTolerance) && (r < hype.fEndOuterRadius) && (r > hype.fEndInnerRadius);
    vecCore::MaskedAssign(safety, !done && cond, sigz);
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    cond = (sigz > kHalfTolerance) && (r > hype.fEndOuterRadius);
    vecCore__MaskedAssignFunc(safety, !done && cond,
                              Sqrt((r - hype.fEndOuterRadius) * (r - hype.fEndOuterRadius) + (sigz) * (sigz)));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    cond = (sigz > kHalfTolerance) && (r < hype.fEndInnerRadius);
    vecCore__MaskedAssignFunc(safety, !done && cond,
                              Sqrt((r - hype.fEndInnerRadius) * (r - hype.fEndInnerRadius) + (sigz) * (sigz)));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    cond =
        (r2 > ((hype.fRmax2 + hype.fTOut2 * absZ * absZ) + kHalfTolerance)) && (absZ > Real_v(0.)) && (absZ < hype.fDz);
    vecCore__MaskedAssignFunc(safety, !done && cond,
                              HypeUtilities::ApproxDistOutside<Real_v>(r, absZ, hype.fRmax, hype.fTOut));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    vecCore__MaskedAssignFunc(safety,
                              !done && (r2 < ((hype.fRmin2 + hype.fTIn2 * absZ * absZ) - kHalfTolerance)) &&
                                  (absZ > Real_v(0.)) && (absZ < hype.fDz),
                              HypeUtilities::ApproxDistInside<Real_v>(r, absZ, hype.fRmin, hype.fTIn2));
  }

  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point,
                                                  Real_v &safety)
  {
    using namespace ::vecgeom::HypeTypes;
    using Bool_v = typename vecCore::Mask_v<Real_v>;
    safety       = 0.;
    Real_v r     = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v absZ  = Abs(point.z());
    Bool_v inside(false), outside(false);
    Bool_v done(false);

    Real_v distZ(0.), distInner(0.), distOuter(0.);
    safety = 0.;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(hype, point, inside, outside);
    done = (!inside && !outside);
    if (vecCore::MaskFull(done)) return;

    vecCore__MaskedAssignFunc(safety, outside, Real_v(-1.0));
    done |= outside;
    if (vecCore::MaskFull(done)) return;

    vecCore__MaskedAssignFunc(distZ, !done && inside, Abs(Abs(point.z()) - hype.fDz));
    if (checkInnerSurfaceTreatment<hypeTypeT>(hype) && hype.fStIn) {
      vecCore__MaskedAssignFunc(distInner, !done && inside,
                                HypeUtilities::ApproxDistOutside<Real_v>(r, absZ, hype.fRmin, hype.fTIn));
    }

    if (checkInnerSurfaceTreatment<hypeTypeT>(hype) && !hype.fStIn) {
      vecCore__MaskedAssignFunc(distInner, !done && inside, (r - hype.fRmin));
    }

    if (!checkInnerSurfaceTreatment<hypeTypeT>(hype) && !hype.fStIn) {
      vecCore__MaskedAssignFunc(distInner, !done && inside, InfinityLength<Real_v>());
    }

    vecCore__MaskedAssignFunc(distOuter, !done && inside,
                              HypeUtilities::ApproxDistInside<Real_v>(r, absZ, hype.fRmax, hype.fTOut2));
    safety = Min(distInner, distOuter);
    safety = Min(safety, distZ);
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
