/*
 * ConeUtilities.h
 *
 *  Created on: June 01, 2017
 *      Author: Raman Sehgal
 */
#ifndef VECGEOM_CONEUTILITIES_H_
#define VECGEOM_CONEUTILITIES_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/ConeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/shapetypes/ConeTypes.h"
#include "VecGeom/volumes/kernel/TubeImplementation.h"
namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedCone;
template <typename T>
struct ConeStruct;
using UnplacedStruct_t = ConeStruct<Precision>;

namespace ConeUtilities {

/**
 * Determine whether a point lies inside a cylindrical sector defined by the
 * two rays that delimit its phi span.
 *
 * The same could be achieved using atan2 to calculate the angle formed
 * by the point, the origin and the X-axes, but this is a lot faster,
 * using only multiplications and comparisons
 *
 * (-x*starty + y*startx) >= 0: calculates whether going from the start vector to the point
 * we are traveling in the CCW direction (taking the shortest direction, of course)
 *
 * (-endx*y + endy*x) >= 0: calculates whether going from the point to the end vector
 * we are traveling in the CCW direction (taking the shortest direction, of course)
 *
 * For a sector smaller than pi, we need that BOTH of them hold true - if going from start, to the
 * point, and then to the end we are travelling in CCW, it's obvious the point is inside the
 * cylindrical sector.
 *
 * For a sector bigger than pi, only one of the conditions needs to be true. This is less obvious why.
 * Since the sector angle is greater than pi, it can be that one of the two vectors might be
 * farther than pi away from the point. In that case, the shortest direction will be CW, so even
 * if the point is inside, only one of the two conditions need to hold.
 *
 * If going from start to point is CCW, then certainly the point is inside as the sector
 * is larger than pi.
 *
 * If going from point to end is CCW, again, the point is certainly inside.
 *
 * The helper can choose the smaller-than-pi versus larger-than-pi test either
 * at compile time or at runtime, depending on what the cone type exposes.
 **/

#if (1)
template <typename Real_v, typename ShapeType, bool onSurfaceT, bool includeSurface = true>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void PointInCyclicalSector(UnplacedStruct_t const &volume,
                                                                               Real_v const &x, Real_v const &y,
                                                                               bool &ret)
{

  using namespace ::vecgeom::ConeTypes;
  // VECGEOM_VALIDATE(SectorType<ShapeType>::value != kNoAngle, << "ShapeType without a
  // sector passed to PointInCyclicalSector");

  // typedef Real_v Real_v;
  // using vecgeom::ConeTypes::SectorType;
  // using vecgeom::ConeTypes::EAngleType;

  Real_v startx = volume.fAlongPhi1x; // GetAlongPhi1X();
  Real_v starty = volume.fAlongPhi1y; // GetAlongPhi1Y();

  Real_v endx = volume.fAlongPhi2x; // GetAlongPhi2X();
  Real_v endy = volume.fAlongPhi2y; // GetAlongPhi2Y();

  bool smallerthanpi;

  if (SectorType<ShapeType>::value == kUnknownAngle)
    smallerthanpi = volume.fDPhi <= M_PI;
  else
    smallerthanpi = SectorType<ShapeType>::value == kOnePi || SectorType<ShapeType>::value == kSmallerThanPi;

  Real_v startCheck = (-x * starty) + (y * startx);
  Real_v endCheck   = (-endx * y) + (endy * x);

  if (onSurfaceT) {
    // in this case, includeSurface is irrelevant
    ret = (Abs(startCheck) <= kConeTolerance) || (Abs(endCheck) <= kConeTolerance);
  } else {
    if (smallerthanpi) {
      if (includeSurface)
        ret = (startCheck >= -kConeTolerance) && (endCheck >= -kConeTolerance);
      else
        ret = (startCheck >= kConeTolerance) && (endCheck >= kConeTolerance);
    } else {
      if (includeSurface)
        ret = (startCheck >= -kConeTolerance) || (endCheck >= -kConeTolerance);
      else
        ret = (startCheck >= kConeTolerance) || (endCheck >= kConeTolerance);
    }
  }
}

#endif

template <typename Real_v, typename ShapeType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ClassifyPointInCyclicalSector(UnplacedStruct_t const &volume,
                                                                                       Real_v const &x, Real_v const &y,
                                                                                       bool &inside, bool &outside)
{
  using namespace ::vecgeom::ConeTypes;

  Real_v startx = volume.fAlongPhi1x;
  Real_v starty = volume.fAlongPhi1y;
  Real_v endx   = volume.fAlongPhi2x;
  Real_v endy   = volume.fAlongPhi2y;

  bool smallerthanpi;
  if (SectorType<ShapeType>::value == kUnknownAngle)
    smallerthanpi = volume.fDPhi <= kPi;
  else
    smallerthanpi = SectorType<ShapeType>::value == kOnePi || SectorType<ShapeType>::value == kSmallerThanPi;

  Real_v startCheck = (-x * starty) + (y * startx);
  Real_v endCheck   = (-endx * y) + (endy * x);
  Real_v zero(0.);
  Real_v tol(kTolerance);

  if (smallerthanpi) {
    inside  = (startCheck > tol) && (endCheck > tol);
    outside = (startCheck < -tol) || (endCheck < -tol);
  } else {
    inside  = (startCheck > tol) || (endCheck > tol);
    outside = (startCheck < -tol) && (endCheck < -tol);
  }

  auto unresolved = !(inside || outside);
  if (!unresolved) return;

  bool onStartSurface = ((x * startx) + (y * starty) >= zero) && (Abs(startCheck) < tol);
  bool onEndSurface   = ((x * endx) + (y * endy) >= zero) && (Abs(endCheck) < tol);
  bool onSurface      = onStartSurface || onEndSurface;

  bool exactOutside = startCheck < zero;
  if (smallerthanpi) {
    exactOutside |= endCheck < zero;
  } else {
    exactOutside &= endCheck < zero;
  }

  outside |= !onSurface && exactOutside;
  inside |= !onSurface && !exactOutside;
}

#if (1)
template <typename Real_v, bool ForInnerRadius>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v GetRadiusOfConeAtPoint(UnplacedStruct_t const &cone,
                                                                                  Real_v const pointZ)
{

  if (ForInnerRadius) {
    if (cone.fRmin1 == cone.fRmin2) {
      return cone.fRmin1;
    } else {
      return cone.fInnerSlope * pointZ + cone.fInnerOffset;
    }

  } else {
    if (cone.fOriginalRmax1 == cone.fOriginalRmax2) {
      return cone.fOriginalRmax1;
    } else {
      return cone.fOuterSlope * pointZ + cone.fOuterOffset;
    }
  }
}

#endif

/**
 * Intersect a trajectory with one cone phi plane.
 *
 * Points on the phi plane lie on `s * (alongX, alongY)`.
 * Points on the trajectory lie on `(x, y) + t * (vx, vy)`.
 * Therefore `s * (alongX, alongY) == (x, y) + t * (vx, vy)`, which gives
 * `t = (alongY * x - alongX * y) / (vy * alongX - vx * alongY)`.
 *
 * For two non-colinear phi planes we also require the hit point to stay on the
 * positive half-line of the chosen phi boundary.
 */

template <typename Real_v, typename ConeType, bool PositiveDirectionOfPhiVector, bool insectorCheck>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void PhiPlaneTrajectoryIntersection(
    Precision alongX, Precision alongY, Precision normalX, Precision normalY, UnplacedStruct_t const &cone,
    Vector3D<Real_v> const &pos, Vector3D<Real_v> const &dir, Real_v &dist, bool &ok)
{
  const Real_v zero(0.0);
  dist = kInfLength;

  // Wedge normals point towards the wedge interior, so the admissible side
  // depends on whether the caller is searching for an entering or exiting hit.
  if (insectorCheck)
    ok = ((dir.x() * normalX) + (dir.y() * normalY) > zero); // DistToIn  -- require tracks entering volume
  else
    ok = ((dir.x() * normalX) + (dir.y() * normalY) < zero); // DistToOut -- require tracks leaving volume
  if (!ok) return;

  Real_v dirDotXY = (dir.y() * alongX) - (dir.x() * alongY);
  if (dirDotXY == 0.) {
    ok = false;
    return;
  }
  dist = ((alongY * pos.x()) - (alongX * pos.y())) / NonZero(dirDotXY);
  if (dist <= -kConeTolerance) {
    ok = false;
    return;
  }

  if (insectorCheck) {
    Real_v hitx          = pos.x() + dist * dir.x();
    Real_v hity          = pos.y() + dist * dir.y();
    Real_v hitz          = pos.z() + dist * dir.z();
    Real_v r2            = (hitx * hitx) + (hity * hity);
    Real_v innerRadIrTol = GetRadiusOfConeAtPoint<Real_v, true>(cone, hitz) + kTolerance;
    Real_v outerRadIrTol = GetRadiusOfConeAtPoint<Real_v, false>(cone, hitz) - kTolerance;

    ok = Abs(hitz) <= cone.fTolIz && (r2 >= innerRadIrTol * innerRadIrTol) && (r2 <= outerRadIrTol * outerRadIrTol);
    if (!ok) return;

    // For cones with two distinct phi planes, keep only intersections on the
    // positive half-line of the chosen boundary ray.
    if (PositiveDirectionOfPhiVector) {
      ok = ((hitx * alongX) + (hity * alongY)) > zero;
    }
  } else {
    if (PositiveDirectionOfPhiVector) {
      Real_v hitx = pos.x() + dist * dir.x();
      Real_v hity = pos.y() + dist * dir.y();
      ok          = ((hitx * alongX) + (hity * alongY)) >= zero;
    }
  }
}

template <typename Real_v, bool ForInnerSurface>
VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> GetNormal(UnplacedStruct_t const &cone, Vector3D<Real_v> const &point)
{

  // typedef Real_v Real_v;
  Real_v rho = point.Perp();
  Vector3D<Real_v> norm(0., 0., 0.);

  if (ForInnerSurface) {
    // Handling inner conical surface
    Precision rmin1 = cone.fRmin1;
    Precision rmin2 = cone.fRmin2;
    if ((rmin1 == rmin2) && (rmin1 != 0.)) {
      // cone act like tube
      norm.Set(-point.x(), -point.y(), 0.);
    } else {
      Precision secRMin = cone.fSecRMin;
      norm.Set(-point.x(), -point.y(), cone.fZNormInner * (rho * secRMin));
    }
  } else {
    Precision rmax1 = cone.fRmax1;
    Precision rmax2 = cone.fRmax2;
    if ((rmax1 == rmax2) && (rmax1 != 0.)) {
      // cone act like tube
      norm.Set(point.x(), point.y(), 0.);
    } else {
      Precision secRMax = cone.fSecRMax;
      norm.Set(point.x(), point.y(), cone.fZNormOuter * (rho * secRMax));
    }
  }
  return norm;
}

template <typename Real_v, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnConicalSurface(UnplacedStruct_t const &cone,
                                                                            Vector3D<Real_v> const &point)
{

  using namespace ConeUtilities;
  using namespace ConeTypes;
  const Real_v rho       = point.Perp2();
  const Real_v coneRad   = GetRadiusOfConeAtPoint<Real_v, ForInnerSurface>(cone, point.z());
  const Real_v coneRad2  = coneRad * coneRad;
  const Real_v tolerance = (ForInnerSurface) ? cone.fInnerTolerance : cone.fOuterTolerance;
  return (rho >= (coneRad2 - tolerance * coneRad)) && (rho <= (coneRad2 + tolerance * coneRad)) &&
         (Abs(point.z()) < (cone.fDz + kConeTolerance));
}

// precondition: point is on cone surface - as returned from IsOnConicalSurface()
template <typename Real_v, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsMovingOutsideConicalSurface(
    UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction)
{
  return direction.Dot(GetNormal<Real_v, ForInnerSurface>(cone, point)) >= Real_v(0.);
}

// precondition: point is on cone surface - as returned from IsOnConicalSurface()
template <typename Real_v, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsMovingInsideConicalSurface(UnplacedStruct_t const &cone,
                                                                                      Vector3D<Real_v> const &point,
                                                                                      Vector3D<Real_v> const &direction)
{
  return direction.Dot(GetNormal<Real_v, ForInnerSurface>(cone, point)) <= Real_v(0.);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnStartPhi(UnplacedStruct_t const &cone,
                                                                      Vector3D<Real_v> const &point)
{
  Real_v startCheck = (-point.x() * cone.fAlongPhi1y) + (point.y() * cone.fAlongPhi1x);
  return ((point.x() * cone.fAlongPhi1x) + (point.y() * cone.fAlongPhi1y) >= Real_v(0.)) &&
         (Abs(startCheck) < Real_v(kTolerance));
}

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnEndPhi(UnplacedStruct_t const &cone,
                                                                    Vector3D<Real_v> const &point)
{
  Real_v endCheck = (-cone.fAlongPhi2x * point.y()) + (cone.fAlongPhi2y * point.x());
  return ((point.x() * cone.fAlongPhi2x) + (point.y() * cone.fAlongPhi2y) >= Real_v(0.)) &&
         (Abs(endCheck) < Real_v(kTolerance));
}

} // namespace ConeUtilities

/* This helper class keeps the generic template path and a scalar specialization side by side. */
template <class Real_v, class coneTypeT>
class ConeHelpers {

public:
  ConeHelpers() {}
  ~ConeHelpers() {}
  template <bool ForDistToIn, bool ForInnerSurface>
  VECCORE_ATT_HOST_DEVICE static bool DetectIntersectionAndCalculateDistanceToConicalSurface(
      UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    const Real_v zero(0.0);

    distance              = kInfLength;
    bool onConicalSurface = IsOnConicalSurface<Real_v, ForInnerSurface>(cone, point);
    if (onConicalSurface) {
      Vector3D<Real_v> normal = ConeUtilities::GetNormal<Real_v, ForInnerSurface>(cone, point);
      if (vecCore::math::Abs(direction.Dot(normal)) == zero) return false;

      bool movingAcrossSurface =
          ForDistToIn ? ConeUtilities::IsMovingInsideConicalSurface<Real_v, ForInnerSurface>(cone, point, direction)
                      : ConeUtilities::IsMovingOutsideConicalSurface<Real_v, ForInnerSurface>(cone, point, direction);

      if (movingAcrossSurface) {
        if (!checkPhiTreatment<coneTypeT>(cone)) {
          distance = zero;
          return true;
        }

        bool insector(false);
        ConeUtilities::PointInCyclicalSector<Real_v, coneTypeT, false, true>(cone, point.x(), point.y(), insector);
        if (insector) {
          distance = zero;
          return true;
        }
      }
    }

    Real_v pDotV2D = point.x() * direction.x() + point.y() * direction.y();

    Real_v a(0.), b(0.), c(0.);
    Precision fDz = cone.fDz;
    if (ForInnerSurface) {

      Precision rmin1 = cone.fRmin1;
      Precision rmin2 = cone.fRmin2;
      if (rmin1 == rmin2) {
        b = pDotV2D;
        a = direction.Perp2();
        c = point.Perp2() - rmin2 * rmin2;
      } else {

        Precision t = cone.fTanInnerApexAngle;
        Real_v newPz(0.);
        if (cone.fRmin2 > cone.fRmin1)
          newPz = (point.z() + fDz + cone.fInnerConeApex) * t;
        else
          newPz = (point.z() - fDz - cone.fInnerConeApex) * t;

        Real_v dirT = direction.z() * t;
        a           = (direction.x() * direction.x()) + (direction.y() * direction.y()) - dirT * dirT;

        b = pDotV2D - (newPz * dirT);
        c = point.Perp2() - (newPz * newPz);
      }

      Real_v b2 = b * b;
      Real_v ac = a * c;
      if (b2 < ac) return false;
      Real_v d2 = b2 - ac;

      Real_v delta = Sqrt(vecCore::math::Abs(d2));
      if (ForDistToIn) {
        if (b >= zero) {
          distance = c / NonZero(-b - delta);
        } else {
          distance = (-b + delta) / NonZero(a);
        }
      } else {
        if (b == zero && delta == zero) return false;
        if (b >= zero) {
          distance = (-b - delta) / NonZero(a);
        } else {
          distance = c / NonZero(-b + delta);
        }
      }

      if (distance < zero) return false;
      Real_v newZ = point.z() + (direction.z() * distance);
      if (Abs(newZ) >= fDz) return false;

    } else {

      // if (rmax1 == rmax2) {
      if (cone.fOriginalRmax1 == cone.fOriginalRmax2) {
        b = pDotV2D;
        a = direction.Perp2();
        c = point.Perp2() - cone.fOriginalRmax2 * cone.fOriginalRmax2;
      } else {

        Precision t = cone.fTanOuterApexAngle;
        Real_v newPz(0.);
        // if (cone.fRmax2 > cone.fRmax1)
        if (cone.fOriginalRmax2 > cone.fOriginalRmax1)
          newPz = (point.z() + fDz + cone.fOuterConeApex) * t;
        else
          newPz = (point.z() - fDz - cone.fOuterConeApex) * t;
        Real_v dirT = direction.z() * t;
        a           = direction.x() * direction.x() + direction.y() * direction.y() - dirT * dirT;
        b           = pDotV2D - (newPz * dirT);
        c           = point.Perp2() - (newPz * newPz);
      }
      Real_v b2 = b * b;
      Real_v ac = a * c;
      if (b2 < ac) return false;
      Real_v d2    = b2 - ac;
      Real_v delta = Sqrt(vecCore::math::Abs(d2));

      if (ForDistToIn) {
        if (b == zero && delta == zero) return false;
        if (b > zero) {
          distance = (-b - delta) / NonZero(a);
        } else {
          distance = c / NonZero(-b + delta);
        }
        Real_v newZ = point.z() + (direction.z() * distance);
        if (Abs(newZ) >= cone.fDz + kHalfTolerance) return false;
      } else {
        if (b < zero) {
          distance = (-b + delta) / NonZero(a);
        } else if (onConicalSurface) {
          distance = (-b - delta) / NonZero(a);
        } else {
          distance = c / NonZero(-b - delta);
        }
        if (distance <= zero) return false;
      }
    }

    if (checkPhiTreatment<coneTypeT>(cone)) {
      Real_v hitx = point.x() + distance * direction.x();
      Real_v hity = point.y() + distance * direction.y();
      bool insector(false);
      ConeUtilities::PointInCyclicalSector<Real_v, coneTypeT, false, true>(cone, hitx, hity, insector);
      if (!insector) return false;
    }
    return true;
  }

  template <bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &completelyinside,
      typename vecCore::Mask_v<Real_v> &completelyoutside)
  {

    typedef typename vecCore::Mask_v<Real_v> Bool_t;

    // very fast check on z-height
    Real_v absz       = Abs(point[2]);
    completelyoutside = absz > MakePlusTolerant<true>(cone.fDz, kConeTolerance);
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<true>(cone.fDz, kConeTolerance);
    }
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }

    // check on RMAX
    Real_v rmax(0.);
    Real_v r2 = point.x() * point.x() + point.y() * point.y();
    // calculate cone radius at the z-height of position
    if (cone.fOriginalRmax1 == cone.fOriginalRmax2)
      rmax = Real_v(cone.fOriginalRmax1);
    else
      rmax = cone.fOuterSlope * point.z() + cone.fOuterOffset;

    completelyoutside |= r2 > MakePlusTolerantSquare<true>(rmax, cone.fOuterTolerance);
    if (ForInside) {
      completelyinside &= r2 < MakeMinusTolerantSquare<true>(rmax, cone.fOuterTolerance);
    }
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }

    // check on RMIN
    if (ConeTypes::checkRminTreatment<coneTypeT>(cone)) {
      Real_v rmin = cone.fInnerSlope * point.z() + cone.fInnerOffset;

      completelyoutside |= r2 < MakeMinusTolerantSquare<true>(rmin, cone.fInnerTolerance);
      if (ForInside) {
        completelyinside &= r2 > MakePlusTolerantSquare<true>(rmin, cone.fInnerTolerance);
      }
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    if (ConeTypes::checkPhiTreatment<coneTypeT>(cone)) {
      Bool_t completelyoutsidephi;
      Bool_t completelyinsidephi;
      cone.fPhiWedge.GenericKernelForContainsAndInside<Real_v, true>(point, completelyinsidephi, completelyoutsidephi);
      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  template <typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &cone,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_v>;
    Bool_v completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<true>(cone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_v(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_v(EInside::kInside));
  }
};

template <class coneTypeT>
class ConeHelpers<Precision, coneTypeT> {

public:
  ConeHelpers() {}
  ~ConeHelpers() {}

  template <bool ForDistToIn, bool ForInnerSurface>
  VECCORE_ATT_HOST_DEVICE static bool DetectIntersectionAndCalculateDistanceToConicalSurface(
      UnplacedStruct_t const &cone, Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
      Precision &distance)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    distance              = kInfLength;
    bool onConicalSurface = IsOnConicalSurface<Precision, ForInnerSurface>(cone, point);

    if (onConicalSurface) {
      // The normal calculation contains a sqrt, so pay it only for the rare
      // surface-start convention path and reuse it for the side test.
      Vector3D<Precision> normal = ConeUtilities::GetNormal<Precision, ForInnerSurface>(cone, point);
      Precision normalDot        = direction.Dot(normal);
      if (vecCore::math::Abs(normalDot) == 0.) return false;

      if (ForDistToIn) {
        bool isMovingInside = normalDot <= 0.;

        if (!checkPhiTreatment<coneTypeT>(cone)) {
          if (isMovingInside) { // && onConicalSurface
            distance = 0.;
            return true;
          }
        } else {
          bool insector(false);
          ConeUtilities::PointInCyclicalSector<Precision, coneTypeT, false, true>(cone, point.x(), point.y(), insector);
          if (insector && isMovingInside) { // && onConicalSurface
            distance = 0.;
            return true;
          }
        }
      }

      else { // !ForDistToIn
        bool isMovingOutside = normalDot >= 0.;

        if (!checkPhiTreatment<coneTypeT>(cone)) {
          if (isMovingOutside) { // && onConicalSurface
            distance = 0.;
            return true;
          }
        } else {
          bool insector(false);
          ConeUtilities::PointInCyclicalSector<Precision, coneTypeT, false, true>(cone, point.x(), point.y(), insector);

          if (insector && isMovingOutside) { // && onConicalSurface
            distance = 0.;
            return true;
          }
        }
      }
    }

    bool ok(false);
    Precision pDotV2D = point.x() * direction.x() + point.y() * direction.y();

    Precision a(kInfLength), b(kInfLength), c(kInfLength);
    if (ForInnerSurface) {

      if (cone.fRmin1 == cone.fRmin2) {
        b = pDotV2D;
        a = direction.Perp2();
        c = point.Perp2() - cone.fRmin2 * cone.fRmin2;
      } else {

        Precision newPz(0.);
        if (cone.fRmin2 > cone.fRmin1)
          newPz = (point.z() + cone.fDz + cone.fInnerConeApex) * cone.fTanInnerApexAngle;
        else
          newPz = (point.z() - cone.fDz - cone.fInnerConeApex) * cone.fTanInnerApexAngle;

        Precision dirT = direction.z() * cone.fTanInnerApexAngle;
        a              = (direction.x() * direction.x()) + (direction.y() * direction.y()) - dirT * dirT;

        b = pDotV2D - (newPz * dirT);
        c = point.Perp2() - (newPz * newPz);
      }

      Precision b2 = b * b;
      Precision ac = a * c;
      if (b2 < ac) return false;

      Precision d2 = b2 - ac;

      Precision delta = Sqrt(d2);
      if (ForDistToIn) {
        if (b >= 0.) {
          distance = (c / NonZero(-b - delta));
        } else {
          distance = (-b + delta) / NonZero(a);
        }
      } else {
        if (b == 0. && delta == 0.) return false;
        if (b >= 0.) {
          distance = (-b - delta) / NonZero(a);
        } else {
          distance = (c / NonZero(-b + delta));
        }
      }

      if (distance < 0.) return false;
      Precision newZ = point.z() + (direction.z() * distance);
      ok             = (Abs(newZ) < cone.fDz);

    } else {

      /*if (cone.fRmax1 == cone.fRmax2) {*/
      if (cone.fOriginalRmax1 == cone.fOriginalRmax2) {

        a = direction.Perp2();
        b = pDotV2D;
        c = (point.Perp2() - cone.fOriginalRmax2 * cone.fOriginalRmax2);
      } else {

        Precision newPz(0.);
        // if (cone.fRmax2 > cone.fRmax1)
        if (cone.fOriginalRmax2 > cone.fOriginalRmax1)
          newPz = (point.z() + cone.fDz + cone.fOuterConeApex) * cone.fTanOuterApexAngle;
        else
          newPz = (point.z() - cone.fDz - cone.fOuterConeApex) * cone.fTanOuterApexAngle;
        Precision dirT = direction.z() * cone.fTanOuterApexAngle;
        a              = direction.x() * direction.x() + direction.y() * direction.y() - dirT * dirT;
        b              = (pDotV2D - (newPz * dirT));
        c              = (point.Perp2() - (newPz * newPz));
      }
      Precision b2 = b * b;
      Precision ac = a * c;
      Precision d2 = b2 - ac;
      if (d2 < 0) return false;
      Precision delta = Sqrt(d2);

      if (ForDistToIn) {
        if (b == 0. && delta == 0.) return false;
        if (b > 0.) {
          distance = (-b - delta) / NonZero(a); // BE ATTENTIVE, not covers the condition for b==0.
        } else {
          distance = (c / NonZero(-b + delta));
        }
        Precision newZ = point.z() + (direction.z() * distance);
        ok             = (Abs(newZ) < cone.fDz + kHalfTolerance);
      } else {
        if (b < 0.) {
          distance = (-b + delta) / NonZero(a);
        } else if (onConicalSurface) {
          distance = (-b - delta) / NonZero(a);
        } else {
          distance = (c / NonZero(-b - delta));
        }
        ok = distance > 0.;
      }

      if (distance < 0.) return false;
    }
    /*   if (distance < 0.) {
         distance = kInfLength;
       }
   */
    if (checkPhiTreatment<coneTypeT>(cone)) {
      Precision hitx(0), hity(0);
      bool insector(false);
      if (distance < kInfLength) {
        hitx = point.x() + distance * direction.x();
        hity = point.y() + distance * direction.y();
      }

      ConeUtilities::PointInCyclicalSector<Precision, coneTypeT, false, true>(cone, hitx, hity, insector);
      ok &= ((insector) && (distance < kInfLength));
    }
    return ok;
  }

  template <bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &cone, Vector3D<Precision> const &point, bool &completelyinside, bool &completelyoutside)
  {

    // very fast check on z-height
    Precision absz    = Abs(point[2]);
    completelyoutside = absz > MakePlusTolerant<ForInside>(cone.fDz, kConeTolerance);
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<ForInside>(cone.fDz, kConeTolerance);
    }
    if (completelyoutside) return;

    // check on RMAX
    Precision r2 = point.x() * point.x() + point.y() * point.y();
    // calculate cone radius at the z-height of position
    Precision rmax = 0.;
    if (cone.fOriginalRmax1 == cone.fOriginalRmax2)
      rmax = cone.fOriginalRmax1;
    else
      rmax = cone.fOuterSlope * point.z() + cone.fOuterOffset;

    completelyoutside |= r2 > MakePlusTolerantSquare<ForInside>(rmax, cone.fOuterTolerance);
    if (ForInside) {
      completelyinside &= r2 < MakeMinusTolerantSquare<ForInside>(rmax, cone.fOuterTolerance);
    }
    if (completelyoutside) return;

    // check on RMIN
    if (ConeTypes::checkRminTreatment<coneTypeT>(cone)) {
      Precision rmin = cone.fInnerSlope * point.z() + cone.fInnerOffset;

      completelyoutside |= r2 <= MakeMinusTolerantSquare<ForInside>(rmin, cone.fInnerTolerance);
      if (ForInside) {
        completelyinside &= r2 > MakePlusTolerantSquare<ForInside>(rmin, cone.fInnerTolerance);
      }
      if (completelyoutside) return;
    }

    if (ConeTypes::checkPhiTreatment<coneTypeT>(cone)) {
      bool completelyoutsidephi(false);
      bool completelyinsidephi(false);
      cone.fPhiWedge.GenericKernelForContainsAndInside<ForInside>(point, completelyinsidephi, completelyoutsidephi);
      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  template <typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &cone,
                                                                  Vector3D<Precision> const &point, Inside_v &inside)
  {
    bool completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<true>(cone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = EInside::kOutside;
    if (completelyinside) inside = EInside::kInside;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
