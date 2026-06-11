/// @file ConeUtilities.h
/// @brief Shared local predicates and distance helpers for cone kernels.
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

/// @brief Return the cached or runtime phi-span category used by sector tests.
/// @tparam ShapeType Cone shape tag carrying the phi-sector category.
/// @param volume Cone data with the runtime phi span for unknown-angle tags.
/// @return True for one-pi and smaller-than-pi sectors.
template <typename ShapeType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsCyclicalSectorSmallerThanPi(UnplacedStruct_t const &volume)
{
  using namespace ::vecgeom::ConeTypes;
  if (SectorType<ShapeType>::value == kUnknownAngle) return volume.fDPhi <= kPi;
  return SectorType<ShapeType>::value == kOnePi || SectorType<ShapeType>::value == kSmallerThanPi;
}

/// @brief Return whether the cone phi sector contains a local `(x,y)` point.
/// @details The predicate uses only the signed cross products against the two
/// phi boundary rays. For spans up to pi both side tests must pass; for wider
/// sectors either side test is sufficient. The tolerance is explicit because
/// distance checks conventionally use `kConeTolerance`, while Inside-consistent
/// handoff checks use `kTolerance`.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ShapeType Cone shape tag carrying the phi-sector category.
/// @tparam onSurfaceT If true, test only whether the point is on a phi plane.
/// @tparam includeSurface If false, require a strict interior sector point.
/// @param volume Cone data with cached phi boundary rays.
/// @param x Local x coordinate.
/// @param y Local y coordinate.
/// @param[out] ret Result of the sector predicate.
/// @param tolerance Signed-side tolerance for the two phi boundary tests.
template <typename Real_v, typename ShapeType, bool onSurfaceT, bool includeSurface = true>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void PointInCyclicalSector(UnplacedStruct_t const &volume,
                                                                               Real_v const &x, Real_v const &y,
                                                                               bool &ret,
                                                                               Precision tolerance = kConeTolerance)
{
  Real_v startx = volume.fAlongPhi1x;
  Real_v starty = volume.fAlongPhi1y;
  Real_v endx   = volume.fAlongPhi2x;
  Real_v endy   = volume.fAlongPhi2y;

  Real_v startCheck = (-x * starty) + (y * startx);
  Real_v endCheck   = (-endx * y) + (endy * x);
  Real_v tol(tolerance);

  if (onSurfaceT) {
    ret = (Abs(startCheck) <= tol) || (Abs(endCheck) <= tol);
  } else {
    bool smallerthanpi = IsCyclicalSectorSmallerThanPi<ShapeType>(volume);
    if (smallerthanpi) {
      if (includeSurface)
        ret = (startCheck >= -tol) && (endCheck >= -tol);
      else
        ret = (startCheck >= tol) && (endCheck >= tol);
    } else {
      if (includeSurface)
        ret = (startCheck >= -tol) || (endCheck >= -tol);
      else
        ret = (startCheck >= tol) || (endCheck >= tol);
    }
  }
}

/// @brief Classify a point against the phi sector using Inside tolerances.
/// @details Points in the tolerance band of a phi plane leave both `inside` and
/// `outside` false so that `Inside` reports `kSurface`.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ShapeType Cone shape tag carrying the phi-sector category.
/// @param volume Cone data with cached phi boundary rays.
/// @param x Local x coordinate.
/// @param y Local y coordinate.
/// @param[out] inside True only for strictly inside sector points.
/// @param[out] outside True only for strictly outside sector points.
template <typename Real_v, typename ShapeType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ClassifyPointInCyclicalSector(UnplacedStruct_t const &volume,
                                                                                       Real_v const &x, Real_v const &y,
                                                                                       bool &inside, bool &outside)
{
  Real_v startx = volume.fAlongPhi1x;
  Real_v starty = volume.fAlongPhi1y;
  Real_v endx   = volume.fAlongPhi2x;
  Real_v endy   = volume.fAlongPhi2y;

  Real_v startCheck = (-x * starty) + (y * startx);
  Real_v endCheck   = (-endx * y) + (endy * x);
  Real_v zero(0.);
  Real_v tol(kTolerance);
  bool smallerthanpi = IsCyclicalSectorSmallerThanPi<ShapeType>(volume);

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

/// @brief Return the inner or outer cone radius at a local z coordinate.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ForInnerRadius Selects `rmin(z)` when true and `rmax(z)` otherwise.
/// @param cone Cone data with cached slopes and offsets.
/// @param pointZ Local z coordinate.
/// @return Interpolated radius for the requested cone side.
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

/// @brief Intersect a trajectory with one cone phi plane.
/// @details Points on the phi plane lie on `s * (alongX, alongY)`. Points on
/// the trajectory lie on `(x, y) + t * (vx, vy)`, which gives
/// `t = (alongY * x - alongX * y) / (vy * alongX - vx * alongY)`. For two
/// non-colinear phi planes the hit must also stay on the positive half-line of
/// the chosen phi boundary.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ConeType Cone shape tag.
/// @tparam PositiveDirectionOfPhiVector Require the hit on the positive phi ray
/// when true.
/// @tparam insectorCheck True for DistanceToIn-style entering checks; false for
/// DistanceToOut-style exiting checks.
/// @param alongX Phi boundary ray x component.
/// @param alongY Phi boundary ray y component.
/// @param normal_x Inward phi-plane normal x component.
/// @param normal_y Inward phi-plane normal y component.
/// @param cone Cone data used for z and radial acceptance.
/// @param pos Local start point.
/// @param dir Local direction.
/// @param[out] dist Distance to the phi-plane hit or `kInfLength`.
/// @param[out] ok True only when the hit satisfies side, z, radial, and ray
/// acceptance checks.
template <typename Real_v, typename ConeType, bool PositiveDirectionOfPhiVector, bool insectorCheck>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void PhiPlaneTrajectoryIntersection(
    Precision alongX, Precision alongY, Precision normal_x, Precision normal_y, UnplacedStruct_t const &cone,
    Vector3D<Real_v> const &pos, Vector3D<Real_v> const &dir, Real_v &dist, bool &ok)
{
  const Real_v zero(0.0);
  dist = kInfLength;

  // Wedge normals point towards the wedge interior, so the admissible side
  // depends on whether the caller is searching for an entering or exiting hit.
  const Real_v normalDot = (dir.x() * normal_x) + (dir.y() * normal_y);
  ok                     = (insectorCheck ? normalDot : -normalDot) > zero;
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

/// @brief Return the normal of the selected conical surface at a local point.
/// @details Inner-surface normals point toward the cone hole and outer-surface
/// normals point away from the solid.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ForInnerSurface Selects the inner conical side when true.
/// @param cone Cone data with cached normal factors.
/// @param point Local point on the requested conical surface.
/// @return Non-normalized outward normal for the requested solid side.
template <typename Real_v, bool ForInnerSurface>
VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> GetNormal(UnplacedStruct_t const &cone, Vector3D<Real_v> const &point)
{
  Real_v rho = point.Perp();
  Vector3D<Real_v> norm(0., 0., 0.);

  if (ForInnerSurface) {
    Precision rmin1 = cone.fRmin1;
    Precision rmin2 = cone.fRmin2;
    if ((rmin1 == rmin2) && (rmin1 != 0.)) {
      norm.Set(-point.x(), -point.y(), 0.);
    } else {
      Precision secRMin = cone.fSecRMin;
      norm.Set(-point.x(), -point.y(), cone.fZNormInner * (rho * secRMin));
    }
  } else {
    Precision rmax1 = cone.fRmax1;
    Precision rmax2 = cone.fRmax2;
    if ((rmax1 == rmax2) && (rmax1 != 0.)) {
      norm.Set(point.x(), point.y(), 0.);
    } else {
      Precision secRMax = cone.fSecRMax;
      norm.Set(point.x(), point.y(), cone.fZNormOuter * (rho * secRMax));
    }
  }
  return norm;
}

/// @brief Check whether a point is on the selected conical surface.
/// @details Uses the same squared-radial tolerance band as cone classification
/// and normal selection so surface-start distance conventions stay aligned with
/// Inside/Contains boundary handling.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ForInnerSurface Selects the inner conical side when true.
/// @param cone Cone data with cached radial tolerances.
/// @param point Local point to test.
/// @return True when the point is within the selected radial tolerance band and
/// inside the z extent.
template <typename Real_v, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnConicalSurface(UnplacedStruct_t const &cone,
                                                                            Vector3D<Real_v> const &point)
{
  const Real_v rho           = point.Perp2();
  const Real_v coneRad       = GetRadiusOfConeAtPoint<Real_v, ForInnerSurface>(cone, point.z());
  const Real_v coneRad2      = coneRad * coneRad;
  const Precision tolerance  = (ForInnerSurface) ? cone.fInnerTolerance : cone.fOuterTolerance;
  const Real_v toleranceBand = Real_v(2. * tolerance) * coneRad;
  return (rho >= coneRad2 - toleranceBand) && (rho <= coneRad2 + toleranceBand) &&
         (Abs(point.z()) < (cone.fDz + kConeTolerance));
}

/// @brief Test whether a surface-start direction leaves the selected cone side.
/// @pre `point` is on the selected conical surface according to
/// `IsOnConicalSurface`.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ForInnerSurface Selects the inner conical side when true.
/// @param cone Cone data with cached normal factors.
/// @param point Local point on the selected conical surface.
/// @param direction Local direction.
/// @return True when the direction has a non-negative dot product with the
/// selected surface normal.
template <typename Real_v, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsMovingOutsideConicalSurface(
    UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction)
{
  return direction.Dot(GetNormal<Real_v, ForInnerSurface>(cone, point)) >= Real_v(0.);
}

/// @brief Test whether a surface-start direction enters through the selected cone side.
/// @pre `point` is on the selected conical surface according to
/// `IsOnConicalSurface`.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam ForInnerSurface Selects the inner conical side when true.
/// @param cone Cone data with cached normal factors.
/// @param point Local point on the selected conical surface.
/// @param direction Local direction.
/// @return True when the direction has a non-positive dot product with the
/// selected surface normal.
template <typename Real_v, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsMovingInsideConicalSurface(UnplacedStruct_t const &cone,
                                                                                      Vector3D<Real_v> const &point,
                                                                                      Vector3D<Real_v> const &direction)
{
  return direction.Dot(GetNormal<Real_v, ForInnerSurface>(cone, point)) <= Real_v(0.);
}

/// @brief Check whether a local point lies on the start-phi boundary ray.
/// @details The point must be on the positive half-line of the boundary ray and
/// within `kTolerance` of its plane.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @param cone Cone data with cached start-phi ray.
/// @param point Local point to test.
/// @return True for points on the start-phi boundary ray.
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnStartPhi(UnplacedStruct_t const &cone,
                                                                      Vector3D<Real_v> const &point)
{
  Real_v startCheck = (-point.x() * cone.fAlongPhi1y) + (point.y() * cone.fAlongPhi1x);
  return ((point.x() * cone.fAlongPhi1x) + (point.y() * cone.fAlongPhi1y) >= Real_v(0.)) &&
         (Abs(startCheck) < Real_v(kTolerance));
}

/// @brief Check whether a local point lies on the end-phi boundary ray.
/// @details The point must be on the positive half-line of the boundary ray and
/// within `kTolerance` of its plane.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @param cone Cone data with cached end-phi ray.
/// @param point Local point to test.
/// @return True for points on the end-phi boundary ray.
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnEndPhi(UnplacedStruct_t const &cone,
                                                                    Vector3D<Real_v> const &point)
{
  Real_v endCheck = (-cone.fAlongPhi2x * point.y()) + (cone.fAlongPhi2y * point.x());
  return ((point.x() * cone.fAlongPhi2x) + (point.y() * cone.fAlongPhi2y) >= Real_v(0.)) &&
         (Abs(endCheck) < Real_v(kTolerance));
}

} // namespace ConeUtilities

/// @brief Shared cone helper kernels for vector and scalar arithmetic paths.
/// @details The generic template path and the scalar specialization intentionally
/// stay side by side because the scalar path carries a few convention-preserving
/// shortcuts that are cheaper than the vector mask machinery.
/// @tparam Real_v Scalar or vector arithmetic type.
/// @tparam coneTypeT Cone shape tag.
template <class Real_v, class coneTypeT>
class ConeHelpers {

public:
  ConeHelpers() {}
  ~ConeHelpers() {}

  /// @brief Find a valid conical-surface intersection for DistanceToIn/Out.
  /// @details Surface starts may return zero distance when the direction crosses
  /// the selected conical side and the phi-sector check accepts the point. Other
  /// candidates are filtered by the quadratic root, z extent, and phi sector.
  /// @tparam ForDistToIn True for entering-distance convention, false for
  /// exiting-distance convention.
  /// @tparam ForInnerSurface Selects the inner conical side when true.
  /// @param cone Cone data with cached slopes, tolerances, and phi state.
  /// @param point Local start point.
  /// @param direction Local direction.
  /// @param[out] distance Accepted distance when the method returns true.
  /// @return True only when a valid conical-surface candidate is found.
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
        ConeUtilities::PointInCyclicalSector<Real_v, coneTypeT, false, true>(cone, point.x(), point.y(), insector,
                                                                             kTolerance);
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

      if (cone.fOriginalRmax1 == cone.fOriginalRmax2) {
        b = pDotV2D;
        a = direction.Perp2();
        c = point.Perp2() - cone.fOriginalRmax2 * cone.fOriginalRmax2;
      } else {

        Precision t = cone.fTanOuterApexAngle;
        Real_v newPz(0.);
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
      ConeUtilities::PointInCyclicalSector<Real_v, coneTypeT, false, true>(cone, hitx, hity, insector, kTolerance);
      if (!insector) return false;
    }
    return true;
  }

  /// @brief Populate strict inside/outside masks for Contains and Inside.
  /// @details Contains uses only the outside mask. Inside also requires the
  /// strict inside mask; points not classified into either mask remain surface.
  /// @tparam ForInside Whether to compute strict-inside information.
  /// @param cone Cone data with cached radial and phi state.
  /// @param point Local point to classify.
  /// @param[out] completelyinside Strict inside mask, meaningful when
  /// `ForInside` is true.
  /// @param[out] completelyoutside Strict outside mask.
  template <bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &cone, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &completelyinside,
      typename vecCore::Mask_v<Real_v> &completelyoutside)
  {

    typedef typename vecCore::Mask_v<Real_v> Bool_t;

    // very fast check on z-height
    constexpr Precision zTolerance = kTolerance;
    Real_v absz                    = Abs(point[2]);
    completelyoutside              = absz > MakePlusTolerant<true>(cone.fDz, zTolerance);
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<true>(cone.fDz, zTolerance);
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

  /// @brief Classify a local point as outside, surface, or inside.
  /// @details The method maps the strict masks from
  /// `GenericKernelForContainsAndInside` to `EInside`; unresolved tolerance-band
  /// points remain `kSurface`.
  /// @tparam Inside_v Classification storage type.
  /// @param cone Cone data with cached radial and phi state.
  /// @param point Local point to classify.
  /// @param[out] inside Classification result.
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

/// @brief Scalar specialization of the shared cone helper kernels.
/// @details Keeps scalar-only branch structure and direct booleans while
/// preserving the same classification contracts as the generic path.
/// @tparam coneTypeT Cone shape tag.
template <class coneTypeT>
class ConeHelpers<Precision, coneTypeT> {

public:
  ConeHelpers() {}
  ~ConeHelpers() {}

  /// @brief Find a valid scalar conical-surface intersection for DistanceToIn/Out.
  /// @details Surface starts may return zero distance when the direction crosses
  /// the selected conical side and the phi-sector check accepts the point. Other
  /// candidates are filtered by the quadratic root, z extent, and phi sector.
  /// @tparam ForDistToIn True for entering-distance convention, false for
  /// exiting-distance convention.
  /// @tparam ForInnerSurface Selects the inner conical side when true.
  /// @param cone Cone data with cached slopes, tolerances, and phi state.
  /// @param point Local start point.
  /// @param direction Local direction.
  /// @param[out] distance Accepted distance when the method returns true.
  /// @return True only when a valid conical-surface candidate is found.
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
          if (isMovingInside) {
            distance = 0.;
            return true;
          }
        } else {
          bool insector(false);
          ConeUtilities::PointInCyclicalSector<Precision, coneTypeT, false, true>(cone, point.x(), point.y(), insector,
                                                                                  kTolerance);
          if (insector && isMovingInside) {
            distance = 0.;
            return true;
          }
        }
      }

      else { // !ForDistToIn
        bool isMovingOutside = normalDot >= 0.;

        if (!checkPhiTreatment<coneTypeT>(cone)) {
          if (isMovingOutside) {
            distance = 0.;
            return true;
          }
        } else {
          bool insector(false);
          ConeUtilities::PointInCyclicalSector<Precision, coneTypeT, false, true>(cone, point.x(), point.y(), insector,
                                                                                  kTolerance);

          if (insector && isMovingOutside) {
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

      if (cone.fOriginalRmax1 == cone.fOriginalRmax2) {

        a = direction.Perp2();
        b = pDotV2D;
        c = (point.Perp2() - cone.fOriginalRmax2 * cone.fOriginalRmax2);
      } else {

        Precision newPz(0.);
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
          distance = (-b - delta) / NonZero(a);
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
    if (checkPhiTreatment<coneTypeT>(cone)) {
      Precision hitx(0), hity(0);
      bool insector(false);
      if (distance < kInfLength) {
        hitx = point.x() + distance * direction.x();
        hity = point.y() + distance * direction.y();
      }

      ConeUtilities::PointInCyclicalSector<Precision, coneTypeT, false, true>(cone, hitx, hity, insector, kTolerance);
      ok &= ((insector) && (distance < kInfLength));
    }
    return ok;
  }

  /// @brief Populate scalar strict inside/outside flags for Contains and Inside.
  /// @details Contains uses only the outside flag. Inside also requires the
  /// strict inside flag; points not classified into either flag remain surface.
  /// @tparam ForInside Whether to compute strict-inside information.
  /// @param cone Cone data with cached radial and phi state.
  /// @param point Local point to classify.
  /// @param[out] completelyinside Strict inside flag, meaningful when
  /// `ForInside` is true.
  /// @param[out] completelyoutside Strict outside flag.
  template <bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &cone, Vector3D<Precision> const &point, bool &completelyinside, bool &completelyoutside)
  {

    // very fast check on z-height
    constexpr Precision zTolerance = kTolerance;
    Precision absz                 = Abs(point[2]);
    completelyoutside              = absz > MakePlusTolerant<true>(cone.fDz, zTolerance);
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<ForInside>(cone.fDz, zTolerance);
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

    completelyoutside |= r2 > MakePlusTolerantSquare<true>(rmax, cone.fOuterTolerance);
    if (ForInside) {
      completelyinside &= r2 < MakeMinusTolerantSquare<ForInside>(rmax, cone.fOuterTolerance);
    }
    if (completelyoutside) return;

    // check on RMIN
    if (ConeTypes::checkRminTreatment<coneTypeT>(cone)) {
      Precision rmin = cone.fInnerSlope * point.z() + cone.fInnerOffset;

      completelyoutside |= r2 <= MakeMinusTolerantSquare<true>(rmin, cone.fInnerTolerance);
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

  /// @brief Classify a scalar local point as outside, surface, or inside.
  /// @details The method maps the strict flags from
  /// `GenericKernelForContainsAndInside` to `EInside`; unresolved tolerance-band
  /// points remain `kSurface`.
  /// @tparam Inside_v Classification storage type.
  /// @param cone Cone data with cached radial and phi state.
  /// @param point Local point to classify.
  /// @param[out] inside Classification result.
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
