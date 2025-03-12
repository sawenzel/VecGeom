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
 * an ordinary (non-templated) implementation of a Boolean solid
 * using the virtual function interface of its constituents
 *
 * TEMPLATE SPECIALIZATION FOR SUBTRACTION
 */
template <>
struct BooleanImplementation<kSubtraction> {
  using PlacedShape_t    = PlacedBooleanVolume<kSubtraction>;
  using UnplacedVolume_t = UnplacedBooleanVolume<kSubtraction>;
  using UnplacedStruct_t = BooleanStruct;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(BooleanStruct const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Vector3D<Real_v> tmp;
    inside = unplaced.fLeftVolume->Contains(point);
    if (vecCore::MaskEmpty(inside)) return;

    auto rightInside = unplaced.fRightVolume->Contains(point);
    inside &= !rightInside;
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(BooleanStruct const &unplaced,
                                                                  Vector3D<Real_v> const &p, Inside_t &inside)
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
      if ((positionA == EInside::kInside && positionB == EInside::kSurface) ||
          (positionB == EInside::kOutside && positionA == EInside::kSurface)
          /*
           ||( positionA == EInside::kSurface && positionB == EInside::kSurface &&
             (   fPtrSolidA->Normal(p) -
               fPtrSolidB->Normal(p) ).mag2() >
             1000.0*G4GeometryTolerance::GetInstance()->GetRadialTolerance() ) )
          */) {
        inside = EInside::kSurface;
        return;
      } else {
        inside = EInside::kOutside;
        return;
      }
    }
    // going to be a bit more complicated due to Surface states
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &p,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    Real_v dist_right, dist_left, advance = 0.;
    Real_v limit = stepMax;
    Vector3D<Real_v> hitpoint(p);
    // check if inside '-'
    auto insideRight = unplaced.fRightVolume->Inside(p) != kOutside;
    // epsil is used to push across boundaries, push records the actual push
    Precision epsil(0.), push(0.);
    while (1) {
      if (insideRight) {
        //    // propagate to outside of '- / RightShape'
        dist_right = unplaced.fRightVolume->PlacedDistanceToOut(hitpoint, dir, limit);
        if (dist_right >= 0.) {
          advance += dist_right + push;
          limit = stepMax - advance;
          epsil = kRelTolerance(hitpoint + dist_right * dir);
          // Push point across the boundary and record push
          hitpoint += (dist_right + epsil) * dir;
          push = epsil;
        } else {
          push = 0.;
        }

        // now master outside 'B'; check if inside 'A'
        if (unplaced.fLeftVolume->Inside(hitpoint) != kOutside) {
          auto check = unplaced.fLeftVolume->PlacedDistanceToOut(hitpoint, dir);
          if (check > epsil) {
            distance = advance;
            return;
          }
        }
      }

      // master outside '-' and outside '+' ;  find distances to both
      dist_left = unplaced.fLeftVolume->DistanceToIn(hitpoint, dir, limit);
      dist_left = vecCore::math::Max(dist_left, 0.);
      if (dist_left >= limit) {
        distance = kInfLength;
        return;
      }

      dist_right = unplaced.fRightVolume->DistanceToIn(hitpoint, dir, limit);
      if (dist_left < dist_right - kTolerance) {
        advance += dist_left + push;
        distance = advance;
        return;
      }

      //        // propagate to '-'
      if (dist_right >= 0. && dist_right < kInfLength) {
        advance += dist_right + push;
        limit = stepMax - advance;
        epsil = kRelTolerance(hitpoint + dist_right * dir);
        hitpoint += (dist_right + epsil) * dir;
        push = epsil;
      } else {
        push = 0.;
      }
      insideRight = true;
    } // end while
  }

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

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(BooleanStruct const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    const auto safetyleft  = unplaced.fLeftVolume->SafetyToOut(point);
    const auto safetyright = unplaced.fRightVolume->SafetyToIn(point);
    safety                 = Min(safetyleft, safetyright);
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(BooleanStruct const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &normal, Bool_v &valid)
  {
    Vector3D<Real_v> localNorm;
    Vector3D<Real_v> localPoint;
    valid = false; // Backend::kFalse;

    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    // If point is inside B, then it must be on a surface of B
    if (fPtrSolidB->Contains(point)) {
      fPtrSolidB->GetTransformation()->Transform(point, localPoint);
      valid = fPtrSolidB->Normal(localPoint, localNorm);
      // The normal to the subtracted solid has to be inverted and transformed back
      localNorm *= -1.;
      fPtrSolidB->GetTransformation()->InverseTransformDirection(localNorm, normal);
      return;
    }

    // If point is outside A, then it must be on a surface of A
    if (!fPtrSolidA->Contains(point)) {
      fPtrSolidA->GetTransformation()->Transform(point, localPoint);
      valid = fPtrSolidA->Normal(localPoint, localNorm);
      fPtrSolidA->GetTransformation()->InverseTransformDirection(localNorm, normal);
      return;
    }

    // Point is inside A and outside B, check safety
    fPtrSolidA->GetTransformation()->Transform(point, localPoint);
    Real_v safetyA = fPtrSolidA->SafetyToOut(localPoint);
    Real_v safetyB = fPtrSolidB->SafetyToIn(point);
    Bool_v onA     = safetyA < safetyB;
    if (vecCore::MaskFull(onA)) {
      valid = fPtrSolidA->Normal(localPoint, localNorm);
      fPtrSolidA->GetTransformation()->InverseTransformDirection(localNorm, normal);
      return;
    } else {
      //  if (vecCore::MaskEmpty(onA)) {  // to use real mask operation when supporting vectors
      fPtrSolidB->GetTransformation()->Transform(point, localPoint);
      valid = fPtrSolidB->Normal(localPoint, localNorm);
      // The normal to the subtracted solid has to be inverted and transformed back
      localNorm *= -1.;
      fPtrSolidB->GetTransformation()->InverseTransformDirection(localNorm, normal);
      return;
    }
    // Some particles are on A, some on B. We never arrive here in the scalar case
    // If the interface to Normal will support the vector case, we have to write code here.
    return;
  }

}; // End struct BooleanImplementation

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

// include stuff for boolean union
#include "BooleanUnionImplementation.h"

// include stuff for boolean intersection
#include "BooleanIntersectionImplementation.h"

#endif /* BOOLEANIMPLEMENTATION_H_ */
