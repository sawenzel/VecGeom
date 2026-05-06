/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANIMPLEMENTATION_H_
#define BOOLEANIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/BooleanStruct.h"
#include <VecCore/VecCore>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1v(struct, BooleanImplementation, BooleanOperation, Arg1);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <BooleanOperation Op>
class PlacedBooleanVolume;
template <BooleanOperation Op>
class UnplacedBooleanVolume;

template <BooleanOperation boolOp>
struct BooleanImplementation {
  using PlacedShape_t    = PlacedBooleanVolume<boolOp>;
  using UnplacedVolume_t = UnplacedBooleanVolume<boolOp>;
  using UnplacedStruct_t = BooleanStruct;

  // empty since functionality will be implemented in
  // partially template specialized structs
};

/**
 * @brief Kernel implementation for Boolean subtraction volumes.
 * @details The implementation represents `left - right` using the virtual
 * interfaces of the two placed constituents. Points in the right constituent
 * are excluded from the result, and normals on the right constituent are
 * inverted because they bound a removed volume.
 */
template <>
struct BooleanImplementation<kSubtraction> {
  using PlacedShape_t    = PlacedBooleanVolume<kSubtraction>;
  using UnplacedVolume_t = UnplacedBooleanVolume<kSubtraction>;
  using UnplacedStruct_t = BooleanStruct;

  /**
   * @brief Test whether a point is contained in the subtraction result.
   * @details The point must be contained by the left constituent and not
   * contained by the subtracted right constituent.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(BooleanStruct const &unplaced,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    inside = unplaced.fLeftVolume->Contains(point);
    if (!inside) return;

    const bool rightInside = unplaced.fRightVolume->Contains(point);
    inside &= !rightInside;
  }

  /**
   * @brief Classify a point with the Boolean subtraction inside convention.
   * @details A point is inside only when it is inside the left constituent and
   * outside the right constituent. Surface cases include the outer left surface
   * and the inner surface introduced by subtracting the right constituent.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(BooleanStruct const &unplaced,
                                                                  Vector3D<Real_v> const &p, vecgeom::Inside_t &inside)
  {

    // now use the Inside functionality of left and right components
    // algorithm taken from Geant4 implementation
    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    const auto positionA = fPtrSolidA->Inside(p);
    if (positionA == EInside::kOutside) {
      inside = EInside::kOutside;
      return;
    }

    const auto positionB = fPtrSolidB->Inside(p);

    if (positionA == EInside::kInside && positionB == EInside::kOutside) {
      inside = EInside::kInside;
      return;
    } else {
      bool surfaceSurfaceBoundary = false;
      if (positionA == EInside::kSurface && positionB == EInside::kSurface) {
        Vector3D<Real_v> normalA, normalB;
        bool validNormalA = fPtrSolidA->Normal(p, normalA);
        bool validNormalB = fPtrSolidB->Normal(p, normalB);
        if (validNormalA && validNormalB) {
          // Different surface normals still define a Boolean boundary.
          if ((normalA - normalB).Mag2() > Real_v(1000.) * kToleranceDist<Real_v>) {
            surfaceSurfaceBoundary = true;
          }
        }
        if (!surfaceSurfaceBoundary) {
          const Real_v zero(0.);
          const Real_v probeStep = Real_v(4.) * kToleranceCone<Real_v>;
          bool foundMaterial     = false;
          bool foundVoid         = false;

          const auto xPlus       = p + Vector3D<Real_v>(probeStep, zero, zero);
          const bool xPlusInside = fPtrSolidA->Contains(xPlus) && !fPtrSolidB->Contains(xPlus);
          foundMaterial |= xPlusInside;
          foundVoid |= !xPlusInside;

          const auto xMinus       = p - Vector3D<Real_v>(probeStep, zero, zero);
          const bool xMinusInside = fPtrSolidA->Contains(xMinus) && !fPtrSolidB->Contains(xMinus);
          foundMaterial |= xMinusInside;
          foundVoid |= !xMinusInside;

          const auto yPlus       = p + Vector3D<Real_v>(zero, probeStep, zero);
          const bool yPlusInside = fPtrSolidA->Contains(yPlus) && !fPtrSolidB->Contains(yPlus);
          foundMaterial |= yPlusInside;
          foundVoid |= !yPlusInside;

          const auto yMinus       = p - Vector3D<Real_v>(zero, probeStep, zero);
          const bool yMinusInside = fPtrSolidA->Contains(yMinus) && !fPtrSolidB->Contains(yMinus);
          foundMaterial |= yMinusInside;
          foundVoid |= !yMinusInside;

          const auto zPlus       = p + Vector3D<Real_v>(zero, zero, probeStep);
          const bool zPlusInside = fPtrSolidA->Contains(zPlus) && !fPtrSolidB->Contains(zPlus);
          foundMaterial |= zPlusInside;
          foundVoid |= !zPlusInside;

          const auto zMinus       = p - Vector3D<Real_v>(zero, zero, probeStep);
          const bool zMinusInside = fPtrSolidA->Contains(zMinus) && !fPtrSolidB->Contains(zMinus);
          foundMaterial |= zMinusInside;
          foundVoid |= !zMinusInside;

          // Equal or unavailable operand normals can be an exact cancellation
          // or just a tolerance-shell section seam. Local Boolean membership
          // distinguishes the two without adding a general Inside call.
          surfaceSurfaceBoundary = foundMaterial && foundVoid;
        }
      }

      if ((positionA == EInside::kInside && positionB == EInside::kSurface) ||
          (positionB == EInside::kOutside && positionA == EInside::kSurface) || surfaceSurfaceBoundary) {
        inside = EInside::kSurface;
        return;
      } else {
        inside = EInside::kOutside;
        return;
      }
    }
    // going to be a bit more complicated due to Surface states
  }

  /**
   * @brief Compute the distance from outside to enter `left - right`.
   * @details The ray alternates between entering the left constituent and
   * exiting the subtracted right constituent. The upfront wrong-side guard is
   * required because a point strictly inside the subtraction result is already
   * in the solid, and the walking algorithm can otherwise clamp the left
   * constituent's negative entry distance into a false zero hit.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &p,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    Real_v dist_right, dist_left, workDistance = 0.;
    Real_v limit                  = stepMax;
    const Real_v overlapTolerance = kToleranceCone<Real_v>;
    Vector3D<Real_v> hitpoint(p);
    // `workDistance`/`hitpoint` include small numerical pushes used to avoid
    // re-hitting the same B surface. `lastBoundaryDistance` is the unpushed
    // geometric boundary just crossed and is the only value returned for such
    // boundary entries.
    Real_v lastBoundaryDistance(0.);
    bool hasPushedBoundary      = false;
    const auto insideLeftState  = unplaced.fLeftVolume->Inside(p);
    const auto insideRightState = unplaced.fRightVolume->Inside(p);
    if (insideLeftState == kInside && insideRightState == kOutside) {
      // A point already inside the subtraction result is on the wrong side for
      // DistanceToIn; do not let the walking algorithm turn A's negative entry
      // distance into a zero hit.
      distance = Real_v(-1.);
      return;
    }
    // check if inside '-'
    auto insideRight = insideRightState != kOutside;
    while (1) {
      if (insideRight) {
        // While inside B, the next possible entry into A-B is the B exit, but
        // only if A continues beyond that exit. Compare the local B exit with
        // the local A exit before applying any push to the working point.
        const auto leftStateBeforeRightExit = unplaced.fLeftVolume->Inside(hitpoint);
        const auto rightStateAtHit          = unplaced.fRightVolume->Inside(hitpoint);
        const Real_v dist_left_out          = leftStateBeforeRightExit != kOutside
                                                  ? unplaced.fLeftVolume->PlacedDistanceToOut(hitpoint, dir, limit)
                                                  : Real_v(kInfLength);
        dist_right                          = unplaced.fRightVolume->PlacedDistanceToOut(hitpoint, dir, limit);

        if (rightStateAtHit == kSurface && dist_right >= Real_v(0.) && dist_right <= overlapTolerance) {
          const Real_v epsil                 = kRelTolerance<Real_v>(hitpoint);
          const auto rightStateAfterZeroExit = unplaced.fRightVolume->Inside(hitpoint + epsil * dir);
          if (rightStateAfterZeroExit == kOutside) {
            // A zero cutter exit from a surface point can be a ray already on the
            // outside side of B. Do not push and keep walking as if still inside
            // the removed volume.
            insideRight = false;
          }
        }

        if (!insideRight) continue;

        if (leftStateBeforeRightExit != kOutside && dist_right >= Real_v(0.) &&
            dist_right < dist_left_out - overlapTolerance) {
          // Exiting B before exiting A is a real entry into A-B. Coincident
          // A/B exits cancel and must be skipped.
          distance = workDistance + dist_right;
          return;
        }

        if (dist_right < Real_v(0.) || dist_right >= limit) {
          if (rightStateAtHit == kSurface) {
            // A B-surface point can already be on the outside side for this
            // ray. Do not force an infinite hit; let the outside-B ordering
            // below decide whether A is entered at zero distance.
            insideRight = false;
          } else {
            distance = kInfLength;
            return;
          }
        } else {
          // The B exit is not a physical entry yet, so record its exact distance
          // and move only the loop state across B. A following zero A hit should
          // return `lastBoundaryDistance`, not the pushed `workDistance`.
          lastBoundaryDistance = workDistance + dist_right;
          const Real_v epsil   = kRelTolerance<Real_v>(hitpoint + dist_right * dir);
          // Push only the working point to avoid seeing the same B surface again;
          // the physical boundary distance is kept separately for returns.
          hitpoint += (dist_right + epsil) * dir;
          workDistance      = lastBoundaryDistance + epsil;
          limit             = stepMax - workDistance;
          hasPushedBoundary = true;
        }
      }

      // Outside B, an A entry is valid only if it happens strictly before the
      // next B entry. Near-coincident A/B entries are treated as canceled or
      // grazing cutter boundaries, not as material entries.
      dist_left = unplaced.fLeftVolume->DistanceToIn(hitpoint, dir, limit);
      dist_left = Max(dist_left, Real_v(0.));
      if (dist_left >= limit) {
        distance = kInfLength;
        return;
      }

      dist_right = unplaced.fRightVolume->DistanceToIn(hitpoint, dir, limit);
      if (dist_left < dist_right - overlapTolerance) {
        // If the previous step just crossed B and the pushed point is already
        // inside A, the zero left distance refers to the B boundary we crossed.
        // Return the geometric boundary, not the internally pushed point.
        distance =
            (hasPushedBoundary && dist_left <= overlapTolerance) ? lastBoundaryDistance : workDistance + dist_left;
        return;
      }
      if (dist_right <= overlapTolerance) {
        // A zero-distance B touch that is not preceded by A is a grazing or
        // canceled A/B boundary; walking it would not make progress, and a
        // later A hit may still be hidden by B.
        distance = kInfLength;
        return;
      }
      if (dist_right >= limit) {
        distance = kInfLength;
        return;
      }

      // B is reached before A, so the ray enters the subtracted volume. Walk
      // the working point across B and continue looking for the following B
      // exit, preserving the exact B-entry distance separately.
      if (dist_right >= 0.) {
        lastBoundaryDistance = workDistance + dist_right;
        const Real_v epsil   = kRelTolerance<Real_v>(hitpoint + dist_right * dir);
        hitpoint += (dist_right + epsil) * dir;
        workDistance      = lastBoundaryDistance + epsil;
        limit             = stepMax - workDistance;
        hasPushedBoundary = true;
      }
      insideRight = true;
    } // end while
  }

  /**
   * @brief Compute the distance from inside `left - right` to leave it.
   * @details The ray leaves the subtraction result either by exiting the left
   * constituent or by entering the removed right constituent, so the result is
   * the smaller of those two distances.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(BooleanStruct const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    const auto distancel  = unplaced.fLeftVolume->PlacedDistanceToOut(point, direction, stepMax);
    const Real_v dinright = unplaced.fRightVolume->DistanceToIn(point, direction, stepMax);
    distance              = Min(distancel, dinright);
    return;
  }

  /**
   * @brief Compute the safety from outside to enter `left - right`.
   * @details The legacy implementation is intentionally approximate: if the
   * point is in both constituents, entry is controlled by leaving the right
   * constituent; otherwise it uses the left constituent SafetyToIn.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(BooleanStruct const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    // very approximate
    if ((fPtrSolidA->Contains(point)) && // case 1
        (fPtrSolidB->Contains(point))) {
      safety = fPtrSolidB->SafetyToOut(fPtrSolidB->GetTransformation()->Transform(point));
    } else {
      // po
      safety = fPtrSolidA->SafetyToIn(point);
    }
  }

  /**
   * @brief Compute the safety from inside `left - right` to leave it.
   * @details The closest exit is either the outer surface of the left
   * constituent or the inner surface of the removed right constituent.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(BooleanStruct const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    // Placed-volume SafetyToOut expects constituent-local coordinates; this
    // also matters for subtraction nodes used as transformed Boolean operands.
    const auto safetyleft =
        unplaced.fLeftVolume->SafetyToOut(unplaced.fLeftVolume->GetTransformation()->Transform(point));
    const auto safetyright = unplaced.fRightVolume->SafetyToIn(point);
    safety                 = Min(safetyleft, safetyright);
  }

  /**
   * @brief Compute an outward normal for the subtraction surface.
   * @details Points on the removed right constituent use that constituent's
   * normal with reversed sign. Points on the left outer boundary use the left
   * normal directly. Ambiguous points inside the left and outside the right are
   * resolved by comparing the two relevant safeties.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &normal, bool &valid)
  {
    valid = false;

    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    // If point is inside B, then it must be on a surface of B
    if (fPtrSolidB->Contains(point)) {
      // VPlacedVolume::Normal expects the point in the Boolean-local frame and
      // performs the constituent transform internally.
      valid = fPtrSolidB->Normal(point, normal);
      // The normal to the subtracted solid has to be inverted.
      normal *= -1.;
      return;
    }

    // If point is outside A, then it must be on a surface of A
    if (!fPtrSolidA->Contains(point)) {
      valid = fPtrSolidA->Normal(point, normal);
      return;
    }

    // Point is inside A and outside B, check safety
    Vector3D<Real_v> localPoint;
    fPtrSolidA->GetTransformation()->Transform(point, localPoint);
    Real_v safetyA = fPtrSolidA->SafetyToOut(localPoint);
    Real_v safetyB = fPtrSolidB->SafetyToIn(point);
    const bool onA = safetyA < safetyB;
    if (onA) {
      valid = fPtrSolidA->Normal(point, normal);
      return;
    } else {
      valid = fPtrSolidB->Normal(point, normal);
      // The normal to the subtracted solid has to be inverted.
      normal *= -1.;
      return;
    }
  }

}; // End struct BooleanImplementation

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

// include stuff for boolean union
#include "BooleanUnionImplementation.h"

// include stuff for boolean intersection
#include "BooleanIntersectionImplementation.h"

#endif /* BOOLEANIMPLEMENTATION_H_ */
