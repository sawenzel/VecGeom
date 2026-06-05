/// @file HypeUtilities.h
/// @brief Helper predicates and root calculations for Hype navigation kernels.
/// @author Raman Sehgal

#ifndef VOLUMES_HYPEUTILITIES_H_
#define VOLUMES_HYPEUTILITIES_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/shapetypes/HypeTypes.h"
#include <cstdio>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedHype;
template <typename T>
struct HypeStruct;

namespace HypeUtilities {
using UnplacedStruct_t = HypeStruct<Precision>;

/// @brief Test whether a point is clearly outside all Hype bounds.
/// @tparam Real_v Floating-point scalar type.
/// @tparam hypeType Hype specialization controlling inner-surface treatment.
/// @param hype Hype runtime data.
/// @param point Local point to test.
/// @return True when z extent, outer radius, or active inner surface rejects the point.
template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE bool IsCompletelyOutside(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point)
{
  using namespace ::vecgeom::HypeTypes;
  Real_v r2    = point.Perp2();
  Real_v oRad2 = (hype.fRmax2 + hype.fTOut2 * point.z() * point.z());

  if (Abs(point.z()) > (hype.fDz + hype.zToleranceLevel)) return true;
  if (r2 > oRad2 + hype.outerRadToleranceLevel) return true;

  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    Real_v iRad2 = (hype.fRmin2 + hype.fTIn2 * point.z() * point.z());
    return r2 < (iRad2 - hype.innerRadToleranceLevel);
  }
  return false;
}

/// @brief Test whether a point is clearly inside all active Hype bounds.
/// @tparam Real_v Floating-point scalar type.
/// @tparam hypeType Hype specialization controlling inner-surface treatment.
/// @param hype Hype runtime data.
/// @param point Local point to test.
/// @return True when the point is separated from all active boundaries by tolerance.
template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE bool IsCompletelyInside(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point)
{
  using namespace ::vecgeom::HypeTypes;
  Real_v r2    = point.Perp2();
  Real_v oRad2 = (hype.fRmax2 + hype.fTOut2 * point.z() * point.z());

  bool completelyinside =
      (Abs(point.z()) < (hype.fDz - hype.zToleranceLevel)) && (r2 < oRad2 - hype.outerRadToleranceLevel);
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    Real_v iRad2 = (hype.fRmin2 + hype.fTIn2 * point.z() * point.z());
    completelyinside &= (r2 > (iRad2 + hype.innerRadToleranceLevel));
  }
  return completelyinside;
}

/// @brief Return squared inner or outer hyperbolic radius at a z coordinate.
/// @tparam Real_v Floating-point scalar type.
/// @tparam ForInnerRad Selects inner radius when true, outer radius when false.
/// @param hype Hype runtime data.
/// @param z Local z coordinate.
/// @return Squared hyperbolic radius.
template <typename Real_v, bool ForInnerRad>
VECCORE_ATT_HOST_DEVICE Real_v RadiusHypeSq(UnplacedStruct_t const &hype, Real_v z)
{

  if (ForInnerRad)
    return (hype.fRmin2 + hype.fTIn2 * z * z);
  else
    return (hype.fRmax2 + hype.fTOut2 * z * z);
}

/// @brief Test whether an outer-surface point moves into material.
/// @tparam Real_v Floating-point scalar type.
/// @param hype Hype runtime data.
/// @param point Local point on the outer hyperbolic surface.
/// @param direction Unit local direction.
/// @return True when the directional derivative points inward beyond tolerance.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointMovingInsideOuterSurface(UnplacedStruct_t const &hype,
                                                             Vector3D<Real_v> const &point,
                                                             Vector3D<Real_v> const &direction)
{
  Real_v pz = point.z();
  Real_v vz = direction.z();
  if (pz < Real_v(0.)) {
    vz = -vz;
    pz = -pz;
  }
  return ((point.x() * direction.x() + point.y() * direction.y() - pz * hype.fTOut2 * vz) < -Real_v(kTolerance));
}

/// @brief Test whether an inner-surface point moves into material.
/// @tparam Real_v Floating-point scalar type.
/// @param hype Hype runtime data.
/// @param point Local point on the inner hyperbolic surface.
/// @param direction Unit local direction.
/// @return True when the directional derivative points away from the hollow region.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointMovingInsideInnerSurface(UnplacedStruct_t const &hype,
                                                             Vector3D<Real_v> const &point,
                                                             Vector3D<Real_v> const &direction)
{
  Real_v pz = point.z();
  Real_v vz = direction.z();

  if (pz < Real_v(0.)) {
    vz = -vz;
    pz = -pz;
  }

  return ((point.x() * direction.x() + point.y() * direction.y() - pz * hype.fTIn2 * vz) > Real_v(kTolerance));
}

/// @brief Test whether a boundary point has an entering direction.
/// @details Checks z caps first, then outer and optional inner hyperbolic
/// surfaces. Direction tests use local surface derivatives rather than a full
/// point classification.
/// @tparam Real_v Floating-point scalar type.
/// @tparam hypeType Hype specialization controlling inner-surface treatment.
/// @param hype Hype runtime data.
/// @param point Local boundary candidate.
/// @param direction Unit local direction.
/// @return True when the point is on an active surface and the ray enters material.
template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE bool IsPointOnSurfaceAndMovingInside(UnplacedStruct_t const &hype,
                                                             Vector3D<Real_v> const &point,
                                                             Vector3D<Real_v> const &direction)
{
  using namespace ::vecgeom::HypeTypes;
  Real_v rho2  = point.Perp2();
  Real_v radI2 = RadiusHypeSq<Real_v, true>(hype, point.z());
  Real_v radO2 = RadiusHypeSq<Real_v, false>(hype, point.z());

  Real_v absZ = Abs(point.z());
  bool zSurf  = ((rho2 - hype.fEndOuterRadius2) < kTolerance) && ((hype.fEndInnerRadius2 - rho2) < kTolerance) &&
                (absZ <= hype.fDz) && (Abs(absZ - hype.fDz) < kTolerance);
  if (zSurf) return point.z() * direction.z() < Real_v(0.);

  bool outerHypeSurf = Abs(radO2 - rho2) < hype.outerRadToleranceLevel;
  if (outerHypeSurf) return IsPointMovingInsideOuterSurface<Real_v>(hype, point, direction);

  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    bool innerHypeSurf = Abs(radI2 - rho2) < hype.innerRadToleranceLevel;
    if (innerHypeSurf) return IsPointMovingInsideInnerSurface<Real_v>(hype, point, direction);
  }
  return false;
}

/// @brief Intersect a ray with the selected z cap and validate the cap annulus.
/// @tparam Real_v Floating-point scalar type.
/// @tparam hypeType Hype specialization controlling inner-surface treatment.
/// @tparam ForDistToIn Selects the cap facing an incoming point when true.
/// @param hype Hype runtime data.
/// @param point Local start point.
/// @param direction Unit local direction.
/// @param[out] zDist Distance to the selected z plane.
/// @return True when the intersection lies in the cap annulus.
template <typename Real_v, typename hypeType, bool ForDistToIn>
VECCORE_ATT_HOST_DEVICE bool GetPointOfIntersectionWithZPlane(UnplacedStruct_t const &hype,
                                                              Vector3D<Real_v> const &point,
                                                              Vector3D<Real_v> const &direction, Real_v &zDist)
{
  using namespace ::vecgeom::HypeTypes;
  zDist = (Sign(ForDistToIn ? point.z() : direction.z()) * hype.fDz - point.z()) / NonZero(direction.z());

  auto r2 = (point + zDist * direction).Perp2();
  // if (!hype.InnerSurfaceExists())
  if (!checkInnerSurfaceTreatment<hypeType>(hype))
    return (r2 < hype.fEndOuterRadius2);
  else
    return ((r2 < hype.fEndOuterRadius2) && (r2 > hype.fEndInnerRadius2));
}

/// @brief Test whether an outer-surface point moves out of material.
/// @tparam Real_v Floating-point scalar type.
/// @param hype Hype runtime data.
/// @param point Local point on the outer hyperbolic surface.
/// @param direction Unit local direction.
/// @return True when the outward derivative is positive beyond tolerance.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointMovingOutsideOuterSurface(UnplacedStruct_t const &hype,
                                                              Vector3D<Real_v> const &point,
                                                              Vector3D<Real_v> const &direction)
{
  Real_v pz = point.z();
  Real_v vz = direction.z();
  if (vz < Real_v(0.)) {
    pz = -pz;
    vz = -vz;
  }
  Vector3D<Real_v> normHere(point.x(), point.y(), -point.z() * hype.fTOut2);
  return normHere.Dot(direction) > Real_v(kTolerance);
}

/// @brief Test whether an inner-surface point moves out of material.
/// @tparam Real_v Floating-point scalar type.
/// @param hype Hype runtime data.
/// @param point Local point on the inner hyperbolic surface.
/// @param direction Unit local direction.
/// @return True when the ray crosses into the hollow region.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointMovingOutsideInnerSurface(UnplacedStruct_t const &hype,
                                                              Vector3D<Real_v> const &point,
                                                              Vector3D<Real_v> const &direction)
{

  Real_v pz = point.z();
  Real_v vz = direction.z();
  if (vz < Real_v(0.)) {
    pz = -pz;
    vz = -vz;
  }
  Vector3D<Real_v> normHere(-point.x(), -point.y(), point.z() * hype.fTIn2);
  return (normHere.Dot(direction) > Real_v(kTolerance));
}

/// @brief Test whether a point on the outer surface has an exiting direction.
/// @tparam Real_v Floating-point scalar type.
/// @param hype Hype runtime data.
/// @param point Local boundary candidate.
/// @param direction Unit local direction.
/// @return True for an outer hyperbolic-surface point moving out of material.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointOnOuterSurfaceAndMovingOutside(UnplacedStruct_t const &hype,
                                                                   Vector3D<Real_v> const &point,
                                                                   Vector3D<Real_v> const &direction)
{
  Real_v rho2        = point.x() * point.x() + point.y() * point.y();
  Real_v absZ        = Abs(point.z());
  Real_v radO2       = RadiusHypeSq<Real_v, false>(hype, point.z());
  bool outerHypeSurf = (Abs(radO2 - rho2) < hype.outerRadToleranceLevel) && (absZ >= Real_v(0.)) && (absZ < hype.fDz);
  return outerHypeSurf && IsPointMovingOutsideOuterSurface<Real_v>(hype, point, direction);
}

/// @brief Test whether a point on the inner surface has an exiting direction.
/// @tparam Real_v Floating-point scalar type.
/// @tparam hypeType Hype specialization controlling inner-surface treatment.
/// @param hype Hype runtime data.
/// @param point Local boundary candidate.
/// @param direction Unit local direction.
/// @return True for an active inner-surface point moving into the hollow region.
template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE bool IsPointOnInnerSurfaceAndMovingOutside(UnplacedStruct_t const &hype,
                                                                   Vector3D<Real_v> const &point,
                                                                   Vector3D<Real_v> const &direction)
{
  using namespace ::vecgeom::HypeTypes;
  Real_v rho2  = point.x() * point.x() + point.y() * point.y();
  Real_v absZ  = Abs(point.z());
  Real_v radI2 = RadiusHypeSq<Real_v, true>(hype, point.z());
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    bool innerHypeSurf = (Abs(radI2 - rho2) < hype.innerRadToleranceLevel) && (absZ >= Real_v(0.)) && (absZ < hype.fDz);
    return innerHypeSurf && HypeUtilities::IsPointMovingOutsideInnerSurface<Real_v>(hype, point, direction);
  }
  return false;
}

/// @brief Test whether a boundary point has an exiting direction.
/// @details Checks z caps first, then outer and optional inner hyperbolic
/// surfaces. Direction tests use local surface derivatives.
/// @tparam Real_v Floating-point scalar type.
/// @tparam hypeType Hype specialization controlling inner-surface treatment.
/// @param hype Hype runtime data.
/// @param point Local boundary candidate.
/// @param direction Unit local direction.
/// @return True when the point is on an active surface and the ray exits material.
template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE bool IsPointOnSurfaceAndMovingOutside(UnplacedStruct_t const &hype,
                                                              Vector3D<Real_v> const &point,
                                                              Vector3D<Real_v> const &direction)
{
  using namespace ::vecgeom::HypeTypes;
  Real_v rho2  = point.x() * point.x() + point.y() * point.y();
  Real_v radI2 = RadiusHypeSq<Real_v, true>(hype, point.z());
  Real_v radO2 = RadiusHypeSq<Real_v, false>(hype, point.z());

  Real_v absZ = Abs(point.z());
  bool zSurf  = ((rho2 - hype.fEndOuterRadius2) < kTolerance) && ((hype.fEndInnerRadius2 - rho2) < kTolerance) &&
                (absZ <= hype.fDz + hype.zToleranceLevel) && (Abs(absZ - hype.fDz) < kTolerance);

  bool out = zSurf && (point.z() * direction.z() > Real_v(0.));
  if (out) return true;

  bool outerHypeSurf = Abs(radO2 - rho2) < hype.outerRadToleranceLevel;
  out                = outerHypeSurf && HypeUtilities::IsPointMovingOutsideOuterSurface<Real_v>(hype, point, direction);
  if (out) return true;

  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    bool innerHypeSurf = Abs(radI2 - rho2) < hype.innerRadToleranceLevel;
    return innerHypeSurf && HypeUtilities::IsPointMovingOutsideInnerSurface<Real_v>(hype, point, direction);
  }

  return false;
}

/// @brief Approximate distance from outside a hyperbolic surface.
/// @tparam Real_v Floating-point scalar type.
/// @param pr Radial coordinate of the query point.
/// @param pz Absolute z coordinate of the query point.
/// @param r0 Radius at z=0.
/// @param tanPhi Tangent of the surface stereo angle.
/// @return Local approximate safety to the surface.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v ApproxDistOutside(Real_v pr, Real_v pz, Precision r0, Precision tanPhi)
{
  Real_v r1 = Sqrt(r0 * r0 + tanPhi * tanPhi * pz * pz);
  Real_v z1 = pz;
  Real_v r2 = pr;
  Real_v z2 = Sqrt((pr * pr - r0 * r0) / (tanPhi * tanPhi));
  Real_v dz = z2 - z1;
  Real_v dr = r2 - r1;
  Real_v r3 = Sqrt(dr * dr + dz * dz);
  return (r3 < vecCore::NumericLimits<Real_v>::Min()) ? (r2 - r1) : (r2 - r1) * dz / r3;
}

/// @brief Approximate distance from inside a hyperbolic surface.
/// @tparam Real_v Floating-point scalar type.
/// @param pr Radial coordinate of the query point.
/// @param pz Absolute z coordinate of the query point.
/// @param r0 Radius at z=0.
/// @param tan2Phi Squared tangent of the surface stereo angle.
/// @return Local approximate safety to the surface.
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v ApproxDistInside(Real_v pr, Real_v pz, Precision r0, Precision tan2Phi)
{
  Real_v tan2Phi_v(tan2Phi);
  if (tan2Phi_v < vecCore::NumericLimits<Real_v>::Min()) return r0 - pr;

  Real_v rh  = Sqrt(r0 * r0 + pz * pz * tan2Phi_v);
  Real_v dr  = -rh;
  Real_v dz  = pz * tan2Phi_v;
  Real_v len = Sqrt(dr * dr + dz * dz);

  return Abs((pr - rh) * dr) / len;
}

} // namespace HypeUtilities

/// @brief Select and validate hyperbolic-surface ray roots.
/// @tparam Real_v Floating-point scalar type.
/// @tparam ForDistToIn Selects entry root ordering when true, exit ordering when false.
/// @tparam ForInnerSurface Selects inner surface when true, outer surface when false.
///
/// @details The helper solves the quadratic for the selected hyperbolic surface,
/// chooses the root matching the entry/exit convention, converts negative roots
/// to infinity, and accepts only roots whose propagated z coordinate remains
/// within the Hype z extent.
template <class Real_v, bool ForDistToIn, bool ForInnerSurface>
class HypeHelpers {

public:
  HypeHelpers() {}
  ~HypeHelpers() {}

  /// @brief Compute the selected hyperbolic-surface ray intersection.
  /// @param hype Hype runtime data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param[out] dist Selected distance or infinity for a rejected negative root.
  /// @return True when a finite quadratic root exists inside the z extent.
  VECCORE_ATT_HOST_DEVICE
  static bool GetPointOfIntersectionWithHyperbolicSurface(HypeStruct<Precision> const &hype,
                                                          Vector3D<Real_v> const &point,
                                                          Vector3D<Real_v> const &direction, Real_v &dist)
  {
    if (ForInnerSurface) {
      Real_v a    = direction.Perp2() - hype.fTIn2 * direction.z() * direction.z();
      Real_v b    = (direction.x() * point.x() + direction.y() * point.y() - hype.fTIn2 * direction.z() * point.z());
      Real_v c    = point.Perp2() - hype.fTIn2 * point.z() * point.z() - hype.fRmin2;
      Real_v disc = b * b - a * c;
      if (!(disc > Real_v(0.))) return false;
      Real_v sqrtDisc = Sqrt(disc);

      if (ForDistToIn)
        dist = (b < Real_v(0.)) ? ((-b + sqrtDisc) / a) : (c / (-b - sqrtDisc));
      else
        dist = (b > Real_v(0.)) ? ((-b - sqrtDisc) / a) : (c / (-b + sqrtDisc));

    } else {
      Real_v a    = direction.Perp2() - hype.fTOut2 * direction.z() * direction.z();
      Real_v b    = (direction.x() * point.x() + direction.y() * point.y() - hype.fTOut2 * direction.z() * point.z());
      Real_v c    = point.Perp2() - hype.fTOut2 * point.z() * point.z() - hype.fRmax2;
      Real_v disc = b * b - a * c;
      if (!(disc > Real_v(0.))) return false;
      Real_v sqrtDisc = Sqrt(disc);

      if (ForDistToIn)
        dist = (b >= Real_v(0.)) ? ((-b - sqrtDisc) / a) : (c / (-b + sqrtDisc));
      else
        dist = (b < Real_v(0.)) ? ((-b + sqrtDisc) / a) : (c / (-b - sqrtDisc));
    }

    if (dist < Real_v(0.)) dist = InfinityLength<Real_v>();

    Real_v newPtZ = point.z() + dist * direction.z();
    return (Abs(newPtZ) <= hype.fDz);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUMES_HYPEUTILITIES_H_ */
