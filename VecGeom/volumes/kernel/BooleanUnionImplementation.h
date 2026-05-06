/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANUNIONIMPLEMENTATION_H_
#define BOOLEANUNIONIMPLEMENTATION_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/BooleanStruct.h"
#include <VecCore/VecCore>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Kernel implementation for Boolean union volumes.
 * @details The implementation delegates geometric queries to the two placed
 * constituent volumes and combines their answers using the union convention:
 * a point belongs to the solid if it belongs to either constituent.
 */
template <>
struct BooleanImplementation<kUnion> {
  using PlacedShape_t    = PlacedBooleanVolume<kUnion>;
  using UnplacedVolume_t = UnplacedBooleanVolume<kUnion>;
  using UnplacedStruct_t = BooleanStruct;

  /**
   * @brief Test whether a point is contained in either constituent.
   * @details The left constituent is queried first and the right query is
   * skipped when the point is already contained in the left constituent.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(BooleanStruct const &unplaced,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    inside = unplaced.fLeftVolume->Contains(point);
    if (inside) return;
    inside |= unplaced.fRightVolume->Contains(point);
  }

  /**
   * @brief Classify a point with the Boolean union inside convention.
   * @details A point strictly inside either constituent is inside the union.
   * Surface points are classified as surface, except for touching constituents
   * with opposite normals, where the local contact is interior to the union.
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
    if (positionA == EInside::kInside) {
      inside = EInside::kInside;
      return;
    }

    const auto positionB = fPtrSolidB->Inside(point);
    if (positionB == EInside::kInside) {
      inside = EInside::kInside;
      return;
    }

    if ((positionA == EInside::kSurface) && (positionB == EInside::kSurface)) {
      Vector3D<Precision> normalA, normalB;
      // VPlacedVolume::Normal expects the Boolean-local point and applies the
      // constituent transform internally. Passing pre-transformed local points
      // double-transforms placed touching constituents.
      fPtrSolidA->Normal(point, normalA);
      fPtrSolidB->Normal(point, normalB);

      if (normalA.Dot(normalB) < 0)
        inside = EInside::kInside; // touching solids -)(-
      else
        inside = EInside::kSurface; // overlapping solids =))
      return;
    } else {
      if ((positionB == EInside::kSurface) || (positionA == EInside::kSurface)) {
        inside = EInside::kSurface;
        return;
      } else {
        inside = EInside::kOutside;
        return;
      }
    }
  }

  /**
   * @brief Compute the distance from outside to enter the union.
   * @details Entering either constituent enters the union, so the result is the
   * minimum of the two constituent DistanceToIn answers.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    const auto d1 = unplaced.fLeftVolume->DistanceToIn(point, direction, stepMax);
    const auto d2 = unplaced.fRightVolume->DistanceToIn(point, direction, stepMax);
    distance      = Min(d1, d2);
  }

  /**
   * @brief Compute the distance from inside the union to leave it.
   * @details The ray may pass from one constituent into the other before
   * leaving the union. The algorithm advances through connected constituents
   * with small pushes across boundaries, then subtracts the final push so the
   * returned distance corresponds to the physical boundary.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(BooleanStruct const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    VPlacedVolume const *const ptrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const ptrSolidB = unplaced.fRightVolume;

    Real_v dist = 0.;
    // Keep the union handoff push close to the surface: kPushTolerance can jump
    // across a nearby connected constituent boundary before the existing
    // connectivity check sees it.
    Real_v pushdist(kTolerance);
    // size_t push          = 0;
    const auto positionA = ptrSolidA->Inside(point);
    Vector3D<Real_v> nextp(point);
    bool connectingstep(false);

    // reusable kernel as lambda
    auto kernel = [&](VPlacedVolume const *A, VPlacedVolume const *B) {
      do {
        connectingstep    = false;
        const auto disTmp = A->PlacedDistanceToOut(nextp, dir);
        dist += (disTmp >= 0. && disTmp < kInfLength) ? disTmp : 0;
        // give a push
        dist += pushdist;
        // push++;
        nextp = point + dist * dir;
        // B could be overlapping with A -- and/or connecting A to another part of A
        // if (B->Contains(nextp)) {
        if (B->Inside(nextp) != vecgeom::kOutside) {
          const auto disTmp = B->PlacedDistanceToOut(nextp, dir);
          dist += (disTmp >= 0. && disTmp < kInfLength) ? disTmp : 0;
          dist += pushdist;
          // push++;
          nextp          = point + dist * dir;
          connectingstep = true;
        }
      } while (connectingstep && (A->Inside(nextp) != kOutside));
    };

    if (positionA != kOutside) { // initially in A
      kernel(ptrSolidA, ptrSolidB);
    }
    // if( positionB != kOutside )
    else {
      kernel(ptrSolidB, ptrSolidA);
    }
    // At the end we need to subtract just one push distance, since intermediate distances
    // from pushed points are smaller than the real distance with the push value
    distance = dist - pushdist;
    if (distance < kTolerance && positionA == kOutside && ptrSolidB->Inside(point) == kOutside) distance = -kTolerance;
    return;
  }

  /**
   * @brief Compute the safety from an outside point to the union.
   * @details The closest way to enter a union is to enter either constituent,
   * so this returns the minimum constituent SafetyToIn. Negative wrong-side
   * values are intentionally preserved for convention checks.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(BooleanStruct const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;
    const auto distA                      = fPtrSolidA->SafetyToIn(point);
    const auto distB                      = fPtrSolidB->SafetyToIn(point);
    safety                                = Min(distA, distB);
    // If safety is negative it should not be made 0 (convention)
  }

  /**
   * @brief Compute the safety from a point in the union to leave it.
   * @details Points outside both constituents are on the wrong side and keep a
   * negative safety. Points inside both constituents must clear both surfaces,
   * while points inside only one constituent use that constituent's SafetyToOut.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(BooleanStruct const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {

    safety                                = -kTolerance; // invalid side
    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    const auto insideA = fPtrSolidA->Inside(point);
    const auto insideB = fPtrSolidB->Inside(point);

    // Is point already outside?
    if (insideA == kOutside && insideB == kOutside) return;

    if (insideA != kOutside && insideB != kOutside) /* in both */
    {
      // Placed-volume SafetyToOut expects constituent-local coordinates for
      // both operands; do not assume the left operand has identity placement.
      safety = Max(fPtrSolidA->SafetyToOut(fPtrSolidA->GetTransformation()->Transform(point)),
                   fPtrSolidB->SafetyToOut(fPtrSolidB->GetTransformation()->Transform(point)));
    } else {
      if (insideA == kSurface || insideB == kSurface) return;
      /* only contained in B */
      if (insideA == kOutside) {
        safety = fPtrSolidB->SafetyToOut(fPtrSolidB->GetTransformation()->Transform(point));
      } else {
        // Placed-volume SafetyToOut expects constituent-local coordinates for
        // both operands; do not assume the left operand has identity placement.
        safety = fPtrSolidA->SafetyToOut(fPtrSolidA->GetTransformation()->Transform(point));
      }
    }
  }

  /**
   * @brief Compute an outward normal for the closest union surface.
   * @details If the point is contained by one constituent, that constituent
   * owns the union surface locally. If the point is outside both constituents,
   * the closest constituent surface is selected using SafetyToIn. Constituent
   * normal calls receive the Boolean-local point because VPlacedVolume applies
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

    // If point is inside A, then it must be on a surface of A (points on the
    // intersection between A and B cannot be on surface, or if they are they
    // are on a common surface and the normal can be computer for A or B)
    if (fPtrSolidA->Contains(point)) {
      // VPlacedVolume::Normal expects the point in the Boolean-local frame and
      // performs the constituent transform internally.
      valid = fPtrSolidA->Normal(point, normal);
      return;
    }
    // Same for points inside B
    if (fPtrSolidB->Contains(point)) {
      valid = fPtrSolidB->Normal(point, normal);
      return;
    }
    // Points outside both A and B can be on any surface. We use the safety.
    const auto safetyA = fPtrSolidA->SafetyToIn(point);
    const auto safetyB = fPtrSolidB->SafetyToIn(point);
    const bool onA     = safetyA < safetyB;
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
