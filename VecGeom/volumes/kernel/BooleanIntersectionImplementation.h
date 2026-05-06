/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANINTERSECTIONIMPLEMENTATION_H_
#define BOOLEANINTERSECTIONIMPLEMENTATION_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/BooleanStruct.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Kernel implementation for Boolean intersection volumes.
 * @details The implementation delegates to the two placed constituents and
 * combines their answers using the intersection convention: a point belongs to
 * the solid only when it belongs to both constituents.
 */
template <>
struct BooleanImplementation<kIntersection> {
  using PlacedShape_t    = PlacedBooleanVolume<kIntersection>;
  using UnplacedVolume_t = UnplacedBooleanVolume<kIntersection>;
  using UnplacedStruct_t = BooleanStruct;

  /**
   * @brief Test whether a point is contained in both constituents.
   * @details Contains is boundary-inclusive for the constituent queries, so a
   * point on a constituent boundary can still be contained by the intersection.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(BooleanStruct const &unplaced,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    const bool insideA = unplaced.fLeftVolume->Contains(point);
    const bool insideB = unplaced.fRightVolume->Contains(point);
    inside             = insideA && insideB;
  }

  /**
   * @brief Classify a point with the Boolean intersection inside convention.
   * @details A point outside either constituent is outside the intersection.
   * It is strictly inside only if both constituents classify it as inside; any
   * inside/surface combination that is not outside is on the intersection
   * surface.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(BooleanStruct const &unplaced,
                                                                  Vector3D<Real_v> const &point,
                                                                  vecgeom::Inside_t &inside)
  {
    // now use the Inside functionality of left and right components
    // algorithm taken from Geant4 implementation
    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    const auto positionA = fPtrSolidA->Inside(point);

    if (positionA == EInside::kOutside) {
      inside = EInside::kOutside;
      return;
    }

    const auto positionB = fPtrSolidB->Inside(point);
    if (positionA == EInside::kInside && positionB == EInside::kInside) {
      inside = EInside::kInside;
      return;
    } else {
      if ((positionA == EInside::kInside && positionB == EInside::kSurface) ||
          (positionB == EInside::kInside && positionA == EInside::kSurface) ||
          (positionA == EInside::kSurface && positionB == EInside::kSurface)) {
        inside = EInside::kSurface;
        return;
      } else {
        inside = EInside::kOutside;
        return;
      }
    }
  }

  /**
   * @brief Compute the distance from outside to enter the intersection.
   * @details The boundary-walking algorithm follows alternating constituent
   * entries until the ray is inside both constituents. A strict-inside guard is
   * kept before the walk because DistanceToIn is a wrong-side query for points
   * already inside the intersection, while surface cases are left to the legacy
   * boundary precheck so entering surface rays still return zero.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    Vector3D<Real_v> hitpoint = point;

    Real_v d1    = 0.;
    Real_v d2    = 0.;
    Real_v snext = 0.0;

    const auto insideLeftState  = unplaced.fLeftVolume->Inside(point);
    const auto insideRightState = unplaced.fRightVolume->Inside(point);
    // For DistanceToIn, constituent surface starts must be re-entered through
    // the normal walking path; otherwise outside approach points on one
    // constituent and inside the other return a false zero for the intersection.
    auto inleft  = insideLeftState == kInside;
    auto inright = insideRightState == kInside;
    if (insideLeftState == kInside && insideRightState == kInside) {
      // Strictly inside both constituents means strictly inside the
      // intersection, so DistanceToIn is called from the wrong side.
      distance = Real_v(-1.0);
      return;
    }

    // just a pre-check before entering main algorithm
    if (inleft && inright) {
      d1 = unplaced.fLeftVolume->PlacedDistanceToOut(hitpoint, dir, stepMax);
      d2 = unplaced.fRightVolume->PlacedDistanceToOut(hitpoint, dir, stepMax);

      // if we are close to a boundary continue
      if (d1 < 2 * kTolerance) inleft = false;
      if (d2 < 2 * kTolerance) inright = false;

      // otherwise exit
      if (inleft && inright) {
        distance = 0.0;
        return;
      }
    }

    // main loop
    while (1) {
      d1 = d2 = 0;
      if (!inleft) {
        d1 = unplaced.fLeftVolume->DistanceToIn(hitpoint, dir);
        // Use only half tolerance for artificial zero-hit progress: it is
        // positive for outside-entry contracts but keeps the reported Boolean
        // entry point on the surface.
        d1 = Max(d1, Real_v(kHalfTolerance));
        if (d1 > 1E20) {
          distance = kInfLength;
          return;
        }
      }
      if (!inright) {
        d2 = unplaced.fRightVolume->DistanceToIn(hitpoint, dir);
        // Keep the same small progress convention for either constituent.
        d2 = Max(d2, Real_v(kHalfTolerance));
        if (d2 > 1E20) {
          distance = kInfLength;
          return;
        }
      }

      if (d1 > d2) {
        // propagate to left shape
        snext += d1;
        inleft = true;
        hitpoint += d1 * dir;

        // check if propagated point is inside right shape
        // check is done with a little push
        inright = unplaced.fRightVolume->Contains(hitpoint + kTolerance * dir);
        if (inright) {
          distance = snext;
          return;
        }
        // here inleft=true, inright=false
      } else {
        // propagate to right shape
        snext += d2;
        inright = true;
        hitpoint += d2 * dir;

        // check if propagated point is inside left shape
        inleft = unplaced.fLeftVolume->Contains(hitpoint + kTolerance * dir);
        if (inleft) {
          distance = snext;
          return;
        }
      }
      // here inleft=false, inright=true
    } // end while loop
    distance = snext;
    return;
  }

  /**
   * @brief Compute the distance from inside the intersection to leave it.
   * @details Leaving either constituent leaves the intersection, so the result
   * is the minimum of the two constituent DistanceToOut answers.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(BooleanStruct const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    distance = Min(unplaced.fLeftVolume->PlacedDistanceToOut(point, direction, stepMax),
                   unplaced.fRightVolume->PlacedDistanceToOut(point, direction, stepMax));
  }

  /**
   * @brief Compute the safety from outside to enter the intersection.
   * @details If the point is already inside one constituent, only the other
   * constituent can block entry. Otherwise the closer constituent entry safety
   * is used, following the Geant4-style approximation kept by the legacy code.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(BooleanStruct const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    // This is the Geant4 algorithm
    // TODO: ROOT seems to produce better safeties
    const auto insideA = unplaced.fLeftVolume->Contains(point);
    const auto insideB = unplaced.fRightVolume->Contains(point);

    if (!insideA && insideB) {
      safety = unplaced.fLeftVolume->SafetyToIn(point);
    } else {
      if (!insideB && insideA) {
        safety = unplaced.fRightVolume->SafetyToIn(point);
      } else {
        safety = Min(unplaced.fLeftVolume->SafetyToIn(point), unplaced.fRightVolume->SafetyToIn(point));
      }
    }
    return;
  }

  /**
   * @brief Compute the safety from inside the intersection to leave it.
   * @details A point outside either constituent is outside the intersection and
   * therefore on the wrong side for SafetyToOut. Otherwise, leaving the closest
   * constituent surface leaves the intersection. Negative constituent values
   * are clamped to zero for the historical inside-side convention.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(BooleanStruct const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    const auto insideA = unplaced.fLeftVolume->Inside(point);
    if (insideA == kOutside) {
      // Outside either constituent is outside the intersection, which is the
      // wrong side for SafetyToOut.
      safety = Real_v(-1.0);
      return;
    }
    const auto insideB = unplaced.fRightVolume->Inside(point);
    if (insideB == kOutside) {
      // Outside either constituent is outside the intersection, which is the
      // wrong side for SafetyToOut.
      safety = Real_v(-1.0);
      return;
    }

    // Placed-volume SafetyToOut expects constituent-local coordinates for both
    // operands; do not assume the left operand has identity placement.
    safety = Min(unplaced.fLeftVolume->SafetyToOut(unplaced.fLeftVolume->GetTransformation()->Transform(point)),
                 unplaced.fRightVolume->SafetyToOut(unplaced.fRightVolume->GetTransformation()->Transform(point)));
    if (safety < Real_v(0.)) safety = Real_v(0.);
  }

  /**
   * @brief Compute an outward normal for the closest intersection surface.
   * @details The selected constituent is the one with the smaller safety to the
   * relevant boundary. `SafetyToOut` follows the unplaced/local-coordinate
   * convention even on placed volumes, while `SafetyToIn` and `Normal` apply
   * the constituent placement internally.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &normal, bool &valid)
  {
    valid = false;

    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;
    Real_v safetyA, safetyB;

    if (fPtrSolidA->Contains(point)) {
      // Placed-volume SafetyToOut expects the point already in constituent-local
      // coordinates; only SafetyToIn and Normal apply placement internally.
      safetyA = fPtrSolidA->SafetyToOut(fPtrSolidA->GetTransformation()->Transform(point));
    } else {
      safetyA = fPtrSolidA->SafetyToIn(point);
    }

    if (fPtrSolidB->Contains(point)) {
      // Placed-volume SafetyToOut expects the point already in constituent-local
      // coordinates; only SafetyToIn and Normal apply placement internally.
      safetyB = fPtrSolidB->SafetyToOut(fPtrSolidB->GetTransformation()->Transform(point));
    } else {
      safetyB = fPtrSolidB->SafetyToIn(point);
    }
    const bool onA = safetyA < safetyB;
    if (onA) {
      valid = fPtrSolidA->Normal(point, normal);
      return;
    } else {
      valid = fPtrSolidB->Normal(point, normal);
      return;
    }
  }
}; // End struct BooleanImplementation

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif /* BooleanImplementation_H_ */
