/*
 * ConeImplementation.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

/// History notes:
/// revision + moving to Vectorized Cone Kernels (Raman Sehgal)
/// May-June 2017: revision + moving to new Structure (Raman Sehgal)
/// 20180323 Guilherme Lima  Adapted to new UnplacedVolume factory

#ifndef VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/shapetypes/ConeTypes.h"
#include "VecGeom/volumes/ConeStruct.h"
#include "VecGeom/volumes/ConeUtilities.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, ConeImplementation, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
class SPlacedCone;
template <typename T>
class SUnplacedCone;

template <typename coneTypeT>
struct ConeImplementation {

  using UnplacedStruct_t = ConeStruct<Precision>;
  using UnplacedVolume_t = SUnplacedCone<coneTypeT>;
  using PlacedShape_t    = SPlacedCone<UnplacedVolume_t>;

  /* Check whether a point already known to lie on a z plane is also on the matching ring edge. */
  template <typename Real_v, bool ForInnerSurface, bool ForLowerZ>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnRing(UnplacedStruct_t const &cone,
                                                                    Vector3D<Real_v> const &point)
  {
    Real_v rad2 = point.Perp2();

    if (ForLowerZ) {
      if (ForInnerSurface) {
        return (rad2 <= MakePlusTolerantSquare<true>(cone.fRmin1)) &&
               (rad2 >= MakeMinusTolerantSquare<true>(cone.fRmin1));
      } else {
        return (rad2 <= MakePlusTolerantSquare<true>(cone.fRmax1)) &&
               (rad2 >= MakeMinusTolerantSquare<true>(cone.fRmax1));
      }
    } else {
      if (ForInnerSurface) {
        return (rad2 <= MakePlusTolerantSquare<true>(cone.fRmin2)) &&
               (rad2 >= MakeMinusTolerantSquare<true>(cone.fRmin2));
      } else {
        return (rad2 <= MakePlusTolerantSquare<true>(cone.fRmax2)) &&
               (rad2 >= MakeMinusTolerantSquare<true>(cone.fRmax2));
      }
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsLeavingOtherBoundaryForPhiRoot(
      UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir, Real_v const &rsq,
      Real_v const &outerRad)
  {
    using namespace ConeUtilities;

    // Suppress a same-boundary phi root only when some other active boundary is
    // already the actual exit direction at the starting edge point.
    bool leavingOtherBoundary = (rsq > MakeMinusTolerantSquare<true>(outerRad, cone.fOuterTolerance)) &&
                                (dir.Dot(GetNormal<Real_v, false>(cone, point)) >= -kHalfTolerance);
    if (ConeTypes::checkRminTreatment<coneTypeT>(cone)) {
      Real_v innerRad     = GetRadiusOfConeAtPoint<Real_v, true>(cone, point.z());
      bool onInnerSurface = rsq < MakePlusTolerantSquare<true>(innerRad, cone.fInnerTolerance);
      leavingOtherBoundary |= onInnerSurface && (dir.Dot(GetNormal<Real_v, true>(cone, point)) >= kHalfTolerance);
    }

    Real_v zSurfaceDistance = Abs(point.z()) - cone.fDz;
    leavingOtherBoundary |= (Abs(zSurfaceDistance) < kConeTolerance) && ((point.z() * dir.z()) > Real_v(0.));
    return leavingOtherBoundary;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnDegenerateClosingRing(UnplacedStruct_t const &cone,
                                                                                     Vector3D<Real_v> const &point)
  {
    using namespace ConeTypes;
    if (!checkRminTreatment<coneTypeT>(cone)) return false;
    if (Abs(Abs(point.z()) - cone.fDz) > Real_v(kTolerance)) return false;

    const bool onUpperEnd = point.z() >= Real_v(0.);
    const Precision rmin  = onUpperEnd ? cone._frmin2 : cone._frmin1;
    const Precision rmax  = onUpperEnd ? cone._frmax2 : cone._frmax1;
    if (vecCore::math::Abs(rmax - rmin) >= kTolerance) return false;

    const Real_v rsq = point.Perp2();
    return (rsq <= MakePlusTolerantSquare<true>(rmax, cone.fOuterTolerance)) &&
           (rsq >= MakeMinusTolerantSquare<true>(rmax, cone.fOuterTolerance));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsEnteringDegenerateClosingRing(
      UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir)
  {
    if (!IsOnDegenerateClosingRing(cone, point)) return false;

    return (dir.Dot(ConeUtilities::GetNormal<Real_v, false>(cone, point)) < -kHalfTolerance) &&
           (dir.Dot(ConeUtilities::GetNormal<Real_v, true>(cone, point)) < -kHalfTolerance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool RejectDegenerateClosingRingDistanceToInCandidate(
      UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir, Real_v const &distance)
  {
    if (!(distance >= Real_v(0.)) || !(distance < Real_v(kInfLength))) return false;

    const Vector3D<Real_v> hit = point + distance * dir;
    return IsOnDegenerateClosingRing(cone, hit) && !IsEnteringDegenerateClosingRing(cone, hit, dir);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &cone,
                                                                    Vector3D<Real_v> const &point,
                                                                    typename vecCore::Mask_v<Real_v> &inside)
  {
    typedef typename vecCore::Mask_v<Real_v> Bool_v;
    Bool_v unused(false);
    Bool_v outside(false);
    ConeHelpers<Real_v, coneTypeT>::template GenericKernelForContainsAndInside<false>(cone, point, unused, outside);
    inside = !outside;
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &cone,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    ConeHelpers<Real_v, coneTypeT>::template Inside<Inside_v>(cone, point, inside);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &cone,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    using namespace ConeUtilities;
    using namespace ConeTypes;
    const Real_v zero(0.);

    // Reject rays that are already outside and moving away before doing any
    // intersection work.
    distance = kInfLength;

    // outside of Z range and going away?
    Real_v distz         = Abs(point.z()) - cone.fDz; // avoid a division for now
    bool outZAndGoingOut = (distz > kTolerance && (point.z() * dir.z()) >= zero) ||
                           (Abs(distz) < kTolerance && (point.z() * dir.z()) > zero);
    if (outZAndGoingOut) return;

    // outside or *on* outer cone and going away?
    Real_v outerRad              = GetRadiusOfConeAtPoint<Real_v, false>(cone, point.z());
    Real_v rsq                   = point.Perp2();
    bool outsideOuterAndGoingOut = (rsq > MakeMinusTolerantSquare<true>(outerRad, cone.fOuterTolerance)) &&
                                   (dir.Dot(GetNormal<Real_v, false>(cone, point)) >= -kHalfTolerance);
    if (outsideOuterAndGoingOut) return;

    //=== Next, check all dimensions of the cone: for points inside --> return -1
    distance = Real_v(-1.0);

    // For points inside z-range, return -1
    bool inside = distz < -kTolerance;

    inside &= rsq < MakeMinusTolerantSquare<true>(outerRad, cone.fOuterTolerance);

    if (checkRminTreatment<coneTypeT>(cone)) {
      Real_v innerRad = GetRadiusOfConeAtPoint<Real_v, true>(cone, point.z());
      inside &= rsq > MakePlusTolerantSquare<true>(innerRad, cone.fInnerTolerance);
    }
    if (checkPhiTreatment<coneTypeT>(cone) && inside) {
      bool completelyinsidephi(false);
      bool completelyoutsidephi(false);
      ClassifyPointInCyclicalSector<Real_v, coneTypeT>(cone, point.x(), point.y(), completelyinsidephi,
                                                       completelyoutsidephi);
      inside &= completelyinsidephi;
    }
    if (inside) return;

    // Check the z-plane candidate before the more expensive conical and phi
    // intersections.
    distance = Real_v(kInfLength);

    if (IsOnDegenerateClosingRing(cone, point) && IsEnteringDegenerateClosingRing(cone, point, dir)) {
      distance = zero;
      return;
    }

    if (distz >= -kTolerance && dir.z() != zero) {
      Real_v distToZ = distz / NonZero(Abs(dir.z()));

#ifdef EDGE_POINTS
      bool onZsurf  = (Abs(point.z()) - cone.fDz) < Real_v(kTolerance);
      bool onLoZSrf = onZsurf && point.z() < zero;
      bool onHiZSrf = onZsurf && point.z() > zero;
      bool loZcond  = onLoZSrf && (IsOnRing<Real_v, false, true>(cone, point));
      bool hiZcond  = onHiZSrf && (IsOnRing<Real_v, false, false>(cone, point));
      if (checkRminTreatment<coneTypeT>(cone)) {
        loZcond |= onLoZSrf && IsOnRing<Real_v, true, true>(cone, point);
        hiZcond |= onHiZSrf && IsOnRing<Real_v, true, false>(cone, point);
      }
      if (loZcond || hiZcond) distToZ = zero;
#endif

      Real_v hitx = point.x() + distToZ * dir.x();
      Real_v hity = point.y() + distToZ * dir.y();

      Real_v r2 = (hitx * hitx) + (hity * hity);

      Precision innerZTol       = cone.fTolIz;
      bool isHittingTopPlane    = (point.z() >= innerZTol) && (r2 <= cone.fSqRmax2 + kTolerance);
      bool isHittingBottomPlane = (point.z() <= -innerZTol) && (r2 <= cone.fSqRmax1 + kTolerance);
      bool okz                  = isHittingTopPlane || isHittingBottomPlane;

      if (checkRminTreatment<coneTypeT>(cone)) {
        isHittingTopPlane &= (r2 >= cone.fSqRmin2 - kTolerance);
        isHittingBottomPlane &= (r2 >= cone.fSqRmin1 - kTolerance);
        okz &= (isHittingTopPlane || isHittingBottomPlane);
      }

      if (checkPhiTreatment<coneTypeT>(cone) && okz) {
        bool insector(false);
        PointInCyclicalSector<Real_v, coneTypeT, false>(cone, hitx, hity, insector);
        okz = insector;
      }
      if (okz && RejectDegenerateClosingRingDistanceToInCandidate(cone, point, dir, distToZ)) okz = false;
      if (okz) {
        distance = distToZ;
        return;
      }
    }

    Real_v dist_rOuter(kInfLength);
    bool ok_outerCone =
        ConeHelpers<Real_v, coneTypeT>::template DetectIntersectionAndCalculateDistanceToConicalSurface<true, false>(
            cone, point, dir, dist_rOuter);
    if (ok_outerCone && RejectDegenerateClosingRingDistanceToInCandidate(cone, point, dir, dist_rOuter)) {
      ok_outerCone = false;
    }
    if (ok_outerCone && dist_rOuter < distance) {
      distance = dist_rOuter;
      return;
    }

    Real_v dist_rInner(kInfLength);
    if (checkRminTreatment<coneTypeT>(cone)) {
      bool ok_innerCone =
          ConeHelpers<Real_v, coneTypeT>::template DetectIntersectionAndCalculateDistanceToConicalSurface<true, true>(
              cone, point, dir, dist_rInner);
      if (ok_innerCone && RejectDegenerateClosingRingDistanceToInCandidate(cone, point, dir, dist_rInner)) {
        ok_innerCone = false;
      }
      if (ok_innerCone && dist_rInner < distance) distance = dist_rInner;
    }

    if (checkPhiTreatment<coneTypeT>(cone)) {
      Real_v startCheck         = (-point.x() * cone.fAlongPhi1y) + (point.y() * cone.fAlongPhi1x);
      Real_v endCheck           = (-cone.fAlongPhi2x * point.y()) + (cone.fAlongPhi2y * point.x());
      bool nearStartPhiBoundary = ((point.x() * cone.fAlongPhi1x) + (point.y() * cone.fAlongPhi1y) >= zero) &&
                                  (Abs(startCheck) <= Real_v(kConeTolerance));
      bool nearEndPhiBoundary   = ((point.x() * cone.fAlongPhi2x) + (point.y() * cone.fAlongPhi2y) >= zero) &&
                                  (Abs(endCheck) <= Real_v(kConeTolerance));
      evolution::Wedge const &w = cone.fPhiWedge;

      Real_v dist_phi(kInfLength);
      bool ok_phi(false);
      PhiPlaneTrajectoryIntersection<Real_v, coneTypeT, SectorType<coneTypeT>::value != kOnePi, true>(
          cone.fAlongPhi1x, cone.fAlongPhi1y, w.GetNormal1().x(), w.GetNormal1().y(), cone, point, dir, dist_phi,
          ok_phi);
      if (ok_phi && dist_phi < distance) {
        bool ignoreStartPhiRoot = false;
        if (nearStartPhiBoundary && dist_phi <= Real_v(kConeTolerance)) {
          ignoreStartPhiRoot = IsLeavingOtherBoundaryForPhiRoot(cone, point, dir, rsq, outerRad);
        }
        if (!ignoreStartPhiRoot) {
          distance = dist_phi;
          return;
        }
      }

      if (SectorType<coneTypeT>::value != kOnePi) {
        PhiPlaneTrajectoryIntersection<Real_v, coneTypeT, true, true>(cone.fAlongPhi2x, cone.fAlongPhi2y,
                                                                      w.GetNormal2().x(), w.GetNormal2().y(), cone,
                                                                      point, dir, dist_phi, ok_phi);
        if (ok_phi && dist_phi < distance) {
          bool ignoreEndPhiRoot = false;
          if (nearEndPhiBoundary && dist_phi <= Real_v(kConeTolerance)) {
            ignoreEndPhiRoot = IsLeavingOtherBoundaryForPhiRoot(cone, point, dir, rsq, outerRad);
          }
          if (!ignoreEndPhiRoot) distance = dist_phi;
        }
      }
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &cone,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /*stepMax*/, Real_v &distance)
  {
    using namespace ConeUtilities;
    using namespace ConeTypes;

    const Real_v zero(0.0);

    // Delay the z division until a z-plane candidate is still plausible.
    Real_v distz = Abs(point.z()) - cone.fDz;

    // Reject points that are already outside before looking for an exit.
    distance = Real_v(-1.0);

    bool outside = distz > Real_v(kConeTolerance);

    Real_v rsq      = point.Perp2();
    Real_v outerRad = ConeUtilities::GetRadiusOfConeAtPoint<Real_v, false>(cone, point.z());
    outside |= rsq > MakePlusTolerantSquare<true>(outerRad, cone.fOuterTolerance);
    if (outside) return;

    // rejection of points on surface and exiting
    const bool onDegenerateClosingRing = IsOnDegenerateClosingRing(cone, point);
    const bool enteringDegenerateClosingRing =
        onDegenerateClosingRing && IsEnteringDegenerateClosingRing(cone, point, direction);
    if (onDegenerateClosingRing && !enteringDegenerateClosingRing) {
      distance = zero;
      return;
    }

    bool onSurfaceAndExiting = (rsq > MakeMinusTolerantSquare<true>(outerRad, cone.fOuterTolerance)) &&
                               (direction.Dot(GetNormal<Real_v, false>(cone, point)) > -kHalfTolerance);
    if (!onDegenerateClosingRing && onSurfaceAndExiting) {
      distance = zero;
      return;
    }

    bool skipRmin = false;
    if (checkRminTreatment<coneTypeT>(cone)) {
      Real_v innerRad = ConeUtilities::GetRadiusOfConeAtPoint<Real_v, true>(cone, point.z());
      if (rsq < MakeMinusTolerantSquare<true>(innerRad, cone.fInnerTolerance)) return;

      // rejection of points on surface and exiting
      bool onInnerSurface = rsq < MakePlusTolerantSquare<true>(innerRad, cone.fInnerTolerance);
      onSurfaceAndExiting = onInnerSurface && (direction.Dot(GetNormal<Real_v, true>(cone, point)) >= kHalfTolerance);
      if (!onDegenerateClosingRing && onSurfaceAndExiting) {
        distance = zero;
        return;
      }
      skipRmin = onInnerSurface;
    }

    if (checkPhiTreatment<coneTypeT>(cone)) {
      bool completelyInsidePhi(false);
      bool completelyOutsidePhi(false);
      ClassifyPointInCyclicalSector<Real_v, coneTypeT>(cone, point.x(), point.y(), completelyInsidePhi,
                                                       completelyOutsidePhi);
      if (completelyOutsidePhi) return;
    }

    bool isGoingUp       = direction.z() > zero;
    bool isGoingDown     = direction.z() < zero;
    bool onUpperZSurface = !onDegenerateClosingRing && point.z() > zero && Abs(distz) < kTolerance;
    bool onLowerZSurface = !onDegenerateClosingRing && point.z() < zero && Abs(distz) < kTolerance;
    bool leavingTopZ     = isGoingUp && onUpperZSurface;
    bool leavingBottomZ  = isGoingDown && onLowerZSurface;
    bool enteringThroughTopZEdge(false);
    bool enteringThroughBottomZEdge(false);
    if (leavingTopZ) {
      bool onOuterConeBoundary = Abs(SafeDistanceToConicalSurface<Real_v, false>(cone, point)) <= cone.fOuterTolerance;
      enteringThroughTopZEdge =
          onOuterConeBoundary && (direction.Dot(GetNormal<Real_v, false>(cone, point)) < -kHalfTolerance);
      if (checkRminTreatment<coneTypeT>(cone)) {
        bool onInnerConeBoundary = Abs(SafeDistanceToConicalSurface<Real_v, true>(cone, point)) <= cone.fInnerTolerance;
        enteringThroughTopZEdge |=
            onInnerConeBoundary && (direction.Dot(GetNormal<Real_v, true>(cone, point)) < -kHalfTolerance);
      }
    }
    if (leavingBottomZ) {
      bool onOuterConeBoundary = Abs(SafeDistanceToConicalSurface<Real_v, false>(cone, point)) <= cone.fOuterTolerance;
      enteringThroughBottomZEdge =
          onOuterConeBoundary && (direction.Dot(GetNormal<Real_v, false>(cone, point)) < -kHalfTolerance);
      if (checkRminTreatment<coneTypeT>(cone)) {
        bool onInnerConeBoundary = Abs(SafeDistanceToConicalSurface<Real_v, true>(cone, point)) <= cone.fInnerTolerance;
        enteringThroughBottomZEdge |=
            onInnerConeBoundary && (direction.Dot(GetNormal<Real_v, true>(cone, point)) < -kHalfTolerance);
      }
    }
    bool enteringThroughZEdge       = enteringThroughTopZEdge || enteringThroughBottomZEdge;
    bool isOnZPlaneAndMovingOutside = !enteringThroughZEdge && (leavingTopZ || leavingBottomZ);
    if (isOnZPlaneAndMovingOutside) {
      distance = distz;
      return;
    }

    // Check the z-plane candidate first; it can terminate the query before the
    // conical and phi-boundary work.
    distance = Real_v(kInfLength);

    Precision fDz  = cone.fDz;
    Real_v dirZInv = Real_v(1.) / NonZero(direction.z());
    if (isGoingUp && !enteringThroughTopZEdge) distance = (fDz - point.z()) * dirZInv;
    if (isGoingDown && !enteringThroughBottomZEdge) distance = (-fDz - point.z()) * dirZInv;

    Real_v dist_rOuter(kInfLength);
    bool ok_outerCone =
        ConeHelpers<Real_v, coneTypeT>::template DetectIntersectionAndCalculateDistanceToConicalSurface<false, false>(
            cone, point, direction, dist_rOuter);
    if (ok_outerCone && dist_rOuter < distance) distance = dist_rOuter;

    Real_v dist_rInner(kInfLength);
    if (checkRminTreatment<coneTypeT>(cone) && !skipRmin) {
      bool ok_innerCone =
          ConeHelpers<Real_v, coneTypeT>::template DetectIntersectionAndCalculateDistanceToConicalSurface<false, true>(
              cone, point, direction, dist_rInner);
      if (ok_innerCone && dist_rInner < distance) distance = dist_rInner;
    }

    if (checkPhiTreatment<coneTypeT>(cone)) {

      bool isOnStartPhi        = ConeUtilities::IsOnStartPhi<Real_v>(cone, point);
      bool isOnEndPhi          = ConeUtilities::IsOnEndPhi<Real_v>(cone, point);
      Vector3D<Real_v> normal1 = cone.fPhiWedge.GetNormal1();
      Vector3D<Real_v> normal2 = cone.fPhiWedge.GetNormal2();
      bool leavingPhiSurface =
          (isOnStartPhi && direction.Dot(-normal1) > zero) || (isOnEndPhi && direction.Dot(-normal2) > zero);
      if (leavingPhiSurface) {
        distance = zero;
        return;
      }

      Real_v dist_phi(kInfLength);
      bool ok_phi               = false;
      evolution::Wedge const &w = cone.fPhiWedge;
      PhiPlaneTrajectoryIntersection<Real_v, coneTypeT, SectorType<coneTypeT>::value != kOnePi, false>(
          cone.fAlongPhi1x, cone.fAlongPhi1y, w.GetNormal1().x(), w.GetNormal1().y(), cone, point, direction, dist_phi,
          ok_phi);
      if (ok_phi && dist_phi < distance) distance = dist_phi;

      if (SectorType<coneTypeT>::value != kOnePi) {
        ConeUtilities::PhiPlaneTrajectoryIntersection<Real_v, coneTypeT, true, false>(
            cone.fAlongPhi2x, cone.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), cone, point, direction,
            dist_phi, ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;
      }
    }
    if (distance < zero && Abs(distance) < kTolerance) distance = zero;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &cone,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace ConeUtilities;
    using namespace ConeTypes;

    safety        = Real_v(-1.0);
    Precision fDz = cone.fDz;
    Real_v distz  = Abs(point.z()) - fDz;

    // For points inside z-range, return -1
    bool inside = distz < -kConeTolerance;

    // This logic to check if the point is inside is far better than
    // using GenericKernel and will improve performance.
    Real_v outerRad = GetRadiusOfConeAtPoint<Real_v, false>(cone, point.z());
    Real_v rsq      = point.Perp2();
    inside &= rsq < MakeMinusTolerantSquare<true>(outerRad, cone.fOuterTolerance);

    if (checkRminTreatment<coneTypeT>(cone)) {
      Real_v innerRad = GetRadiusOfConeAtPoint<Real_v, true>(cone, point.z());
      inside &= rsq > MakePlusTolerantSquare<true>(innerRad, cone.fInnerTolerance);
    }
    Real_v safetyPhi(-kInfLength);
    if (checkPhiTreatment<coneTypeT>(cone) && inside) {
      safetyPhi = cone.fPhiWedge.SafetyToIn(point);
      inside    = safetyPhi < Real_v(0.);
    }
    if (inside) return;

    // Once it is checked that the point is inside or not, safety can be set to 0.
    // This will serve the case that the point is on the surface. So no need to check
    // that the point is really on surface.
    safety = Real_v(0.);

    // Now if the point is neither inside nor on surface, then it should be outside
    // and the safety should be set to some finite value, which is done by below logic

    Real_v safeZ                = Abs(point.z()) - fDz;
    Real_v safeDistOuterSurface = -SafeDistanceToConicalSurface<Real_v, false>(cone, point);

    Real_v safeDistInnerSurface(-kInfLength);
    if (checkRminTreatment<coneTypeT>(cone)) {
      safeDistInnerSurface = -SafeDistanceToConicalSurface<Real_v, true>(cone, point);
    }

    safety = Max(safeZ, Max(safeDistOuterSurface, safeDistInnerSurface));

    if (checkPhiTreatment<coneTypeT>(cone)) {
      safety = Max(safetyPhi, safety);
    }

    if (vecCore::math::Abs(safety) < kTolerance) safety = Real_v(0.);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &cone,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;

    safety = Real_v(-1.0);

    Real_v distz = Abs(point.z()) - cone.fDz;
    Real_v rsq   = point.Perp2();

    // This logic to check if the point is outside is far better then
    // using GenericKernel and will improve performance.
    bool outside = distz > Real_v(kConeTolerance);

    Real_v outerRad = GetRadiusOfConeAtPoint<Real_v, false>(cone, point.z());
    outside |= rsq > MakePlusTolerantSquare<true>(outerRad, cone.fOuterTolerance);

    if (checkRminTreatment<coneTypeT>(cone)) {
      Real_v innerRad = GetRadiusOfConeAtPoint<Real_v, true>(cone, point.z());
      outside |= rsq < MakeMinusTolerantSquare<true>(innerRad, cone.fInnerTolerance);
    }

    Real_v safetyPhi(kInfLength);
    if (checkPhiTreatment<coneTypeT>(cone) && !outside) {
      safetyPhi = cone.fPhiWedge.SafetyToOut(point);
      outside |= safetyPhi < Real_v(0.);
    }
    if (outside) return;

    // Once it is checked that the point is inside or not, safety can be set to 0.
    // This will serve the case that the point is on the surface. So no need to check
    // that the point is really on surface.
    safety = Real_v(0.);

    // Now if the point is neither outside nor on surface, then it should be inside
    // and the safety should be set to some finite value, which is done by below logic

    Precision fDz = cone.fDz;
    Real_v safeZ  = fDz - Abs(point.z());

    Real_v safeDistOuterSurface = SafeDistanceToConicalSurface<Real_v, false>(cone, point);
    Real_v safeDistInnerSurface(kInfLength);
    if (checkRminTreatment<coneTypeT>(cone)) {
      safeDistInnerSurface = SafeDistanceToConicalSurface<Real_v, true>(cone, point);
    }

    safety = Min(safeZ, Min(safeDistOuterSurface, safeDistInnerSurface));

    if (checkPhiTreatment<coneTypeT>(cone)) {
      safety = Min(safetyPhi, safety);
    }
    if (vecCore::math::Abs(safety) < kTolerance) safety = Real_v(0.);
  }

  template <typename Real_v, bool ForInnerSurface>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v SafeDistanceToConicalSurface(UnplacedStruct_t const &cone,
                                                                                          Vector3D<Real_v> const &point)
  {

    typedef Real_v Float_t;
    Float_t rho = point.Perp();
    if (ForInnerSurface) {
      Float_t pRMin = cone.fTanRMin * point.z() + (cone.fRmin1 + cone.fRmin2) * Float_t(0.5); // cone.fRminAv;
      return (rho - pRMin) * cone.fInvSecRMin;
    } else {
      Float_t pRMax = cone.fTanRMax * point.z() + (cone.fRmax1 + cone.fRmax2) * Float_t(0.5); // cone.fRmaxAv;
      return (pRMax - rho) * cone.fInvSecRMax;
    }
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
