// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file volumes/kernel/EllipticalConeImplementation.h
/// @brief Navigation kernels for the elliptical cone primitive.
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipticalConeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>
#include <iomanip>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipticalConeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipticalConeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipticalCone;
template <typename T>
struct EllipticalConeStruct;
class UnplacedEllipticalCone;

/// @brief Implements classification, distance, safety, and normal queries for an elliptical cone.
/// @details The lateral surface is evaluated in scaled cone coordinates,
/// `(x/dx)^2 + (y/dy)^2 = (z - dz)^2`, and intersected with the two z cuts.
/// Distance kernels solve the resulting quadratic and then clip the accepted
/// interval against the z planes and the physical lower nappe.
struct EllipticalConeImplementation {

  using PlacedShape_t    = PlacedEllipticalCone;
  using UnplacedStruct_t = EllipticalConeStruct<Precision>;
  using UnplacedVolume_t = UnplacedEllipticalCone;

  /// @brief Tests whether a point is not outside the elliptical cone.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @tparam Bool_t Scalar boolean output type.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param[out] inside Set to true for inside or surface points.
  template <typename Real_v, typename Bool_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &ellipticalcone,
                                                                    Vector3D<Real_v> const &point, Bool_t &inside)
  {
    bool unused = false, outside = false;
    GenericKernelForContainsAndInside<Real_v, bool, false>(ellipticalcone, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classifies a point against the elliptical cone.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @tparam Inside_t Scalar inside-code output type.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param[out] inside Set to `kInside`, `kSurface`, or `kOutside`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &ellipticalcone,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside = false, completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, bool, true>(ellipticalcone, point, completelyinside, completelyoutside);
    inside = Inside_t(EInside::kSurface);
    if (completelyoutside) inside = Inside_t(EInside::kOutside);
    if (completelyinside) inside = Inside_t(EInside::kInside);
  }

  /// @brief Shared scalar classification predicate for `Contains` and `Inside`.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @tparam Bool_t Scalar boolean output type.
  /// @tparam ForInside Whether the inside classification band is required.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param[out] completelyinside Set when the point is inside beyond tolerance.
  /// @param[out] completelyoutside Set when the point is outside beyond tolerance.
  template <typename Real_v, typename Bool_t, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, Bool_t &completelyinside,
      Bool_t &completelyoutside)
  {
    Real_v px     = point.x() * ellipticalcone.invDx;
    Real_v py     = point.y() * ellipticalcone.invDy;
    Real_v pz     = point.z();
    Real_v hp     = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v ds     = (hp - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    Real_v dz     = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    Real_v safety = vecCore::math::Max(ds, dz);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  /// @brief Computes the distance from an outside point to the cone boundary.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param direction Unit direction in local coordinates.
  /// @param stepMax Unused step limit kept for the common volume interface.
  /// @param[out] distance Entry distance, or `kInfLength` when no forward entry is found.
  /// @details The method solves the lateral quadratic in scaled coordinates and
  /// clips the candidate interval to the z cuts. Far points moving toward the
  /// cone are translated closer to the bounding sphere before solving to reduce
  /// roundoff, and the reported distance includes that offset.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &ellipticalcone,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    Real_v kTwoEpsilon = 2. * kEpsilon;
    distance           = Real_v(kInfLength);
    Real_v offset(0.);
    Vector3D<Real_v> p(point);

    // Move point closer, if required
    Real_v Rfar2(1024. * ellipticalcone.fRsph * ellipticalcone.fRsph); // 1024 = 32 * 32
    if ((p.Mag2() > Rfar2) && (direction.Dot(p) < Real_v(0.))) offset = p.Mag() - Real_v(2.) * ellipticalcone.fRsph;
    p += offset * direction;

    // Special cases to keep in mind:
    //   0) Point is on the surface and leaving the solid
    //   1) Trajectory is parallel to the surface (A = 0, single root at t = -C/2B)
    //   2) No intersection (D < 0) or touch (D < eps) with lateral surface
    //   3) Exception: when the trajectory traverses the apex (D < eps) and A < 0
    //      then always there is an intersection with the solid

    // Set working variables, transform elliptical cone to cone
    Real_v px  = p.x() * ellipticalcone.invDx;
    Real_v py  = p.y() * ellipticalcone.invDy;
    Real_v pz  = p.z();
    Real_v pz0 = p.z() - ellipticalcone.fDz; // pz if apex would be in origin
    Real_v vx  = direction.x() * ellipticalcone.invDx;
    Real_v vy  = direction.y() * ellipticalcone.invDy;
    Real_v vz  = direction.z();

    // Compute coefficients of the quadratic equation: A t^2 + 2B t + C = 0
    Real_v Ar = vx * vx + vy * vy;
    Real_v Br = px * vx + py * vy;
    Real_v Cr = px * px + py * py;
    // 1) Check if A = 0
    // If so, slightly modify vz to avoid degeneration of the quadratic equation
    // The magnitude of vz will be modified in a way that preserves correct behavior when 0) point is leaving the solid
    Real_v vzvz = vz * vz;
    bool tinyA  = vecCore::math::Abs(Ar - vzvz) < kTwoEpsilon * vzvz;
    if (tinyA) vz += vecCore::math::Abs(vz) * kTwoEpsilon;

    Real_v Az = vz * vz;
    Real_v Bz = pz0 * vz;
    Real_v Cz = pz0 * pz0;
    Real_v A  = Ar - Az;
    Real_v B  = Br - Bz;
    Real_v B0 = Br - pz0 * direction.z(); // B calculated with original v.z()
    Real_v C  = Cr - Cz;
    Real_v D  = B * B - A * C;

    // 0) Check if point is leaving the solid
    Real_v sfz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    Real_v nz  = vecCore::math::Sqrt(Cr);
    Real_v sfr = (nz + pz0) * ellipticalcone.cosAxisMin;
    if (vecCore::math::Abs(p.x()) + vecCore::math::Abs(p.y()) < Real_v(0.1) * kHalfTolerance) nz = Real_v(1.);
    Real_v pzA           = pz0 + ellipticalcone.dApex; // slightly shifted apex position for "flying away" check
    Real_v lateralMotion = Br + nz * vz;
    bool done =
        (sfz >= -kHalfTolerance && pz * vz >= Real_v(0.)) || (sfr >= -kHalfTolerance && lateralMotion >= Real_v(0.)) ||
        (pz0 * ellipticalcone.cosAxisMin > -kHalfTolerance && (Cr - pzA * pzA) <= Real_v(0.) && A >= Real_v(0.));
    // lateralMotion is nz times the lateral-surface derivative; scale the tolerance by the same factor.
    done |= sfz <= kHalfTolerance && vecCore::math::Abs(sfr) <= kHalfTolerance && lateralMotion >= -kHalfTolerance * nz;

    // 2) Check if scratching (D < eps & A > 0) or no intersection (D < 0)
    // 3) if (D < eps & A < 0) then trajectory traverses the apex area - continue calculation
    if (sfr <= Real_v(0.) && D < Real_v(0.)) D = Real_v(0.);
    done |= (D < Real_v(0.)) || ((D < kTwoEpsilon * B * B) && (A >= Real_v(0.)));
    if (done) return;

    // Find intersection with Z planes
    Real_v invz  = Real_v(-1.) / NonZero(vz);
    Real_v dz    = vecCore::math::CopySign(Real_v(ellipticalcone.fZCut), invz);
    Real_v tzin  = (pz - dz) * invz;
    Real_v tzout = (pz + dz) * invz;

    // Find roots of the quadratic equation
    Real_v tmp = -B - vecCore::math::CopySign(vecCore::math::Sqrt(D), B);
    Real_v t1  = tmp / A;
    Real_v t2(0.);
    if (tmp != Real_v(0.)) t2 = C / tmp;
    if (tinyA && B != Real_v(0.)) t2 = -C / (Real_v(2.) * B0); // A ~ 0, t = -C / 2B
    Real_v tmin = vecCore::math::Min(t1, t2);
    Real_v tmax = vecCore::math::Max(t1, t2);

    // Set default - intersection with lower nappe (A > 0)
    Real_v trin  = tmin;
    Real_v trout = tmax;
    // Check if intersection with upper nappe only, return infinity
    done |= (A >= Real_v(0.) && pz0 + vz * tmin >= Real_v(0.));
    if (done) return;

    // Check if intersection with both nappes (A < 0)
    if (A < Real_v(0.)) {
      trin  = Real_v(-kInfLength);
      trout = Real_v(kInfLength);
      if (vz < Real_v(0.)) trin = tmax;
      if (vz > Real_v(0.)) trout = tmin;
    }

    // Set distance
    // No special check for inside points, distance for inside points will be negative
    Real_v tin  = vecCore::math::Max(tzin, trin);
    Real_v tout = vecCore::math::Min(tzout, trout);
    if ((tout - tin) >= kHalfTolerance) distance = tin + offset;
  }

  /// @brief Computes the distance from an inside point to leave the cone.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param direction Unit direction in local coordinates.
  /// @param stepMax Unused step limit kept for the common volume interface.
  /// @param[out] distance Exit distance, `0` for tolerated no-exit/tangent cases, or `-1` for outside starts.
  /// @details The lateral quadratic is clipped against the two z planes. The
  /// upper nappe is rejected because it is not part of the physical solid; when
  /// both nappes are intersected, only the forward lower-nappe exit candidate is used.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &ellipticalcone,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    Real_v kTwoEpsilon = 2. * kEpsilon;
    distance           = Real_v(0.);

    // Special cases to keep in mind:
    //   0) Point is on the surface and leaving the solid
    //   1) Trajectory is parallel to the surface (A = 0, single root at t = -C/2B)
    //   2) No intersection (D < 0) or touch (D < eps) with lateral surface
    //   3) Exception: when the trajectory traverses the apex (D < eps) and A < 0
    //      then always there is an intersection with the solid

    // Set working variables, transform elliptical cone to cone
    Real_v px  = point.x() * ellipticalcone.invDx;
    Real_v py  = point.y() * ellipticalcone.invDy;
    Real_v pz  = point.z();
    Real_v pz0 = pz - ellipticalcone.fDz; // pz if apex would be in origin
    Real_v hp  = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v sfr = (hp - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    Real_v sfz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;

    // Check if point is outside
    bool outside = vecCore::math::Max(sfr, sfz) > kHalfTolerance;
    if (outside) {
      distance = Real_v(-1.);
      return;
    }

    // Compute coefficients of the quadratic equation: A t^2 + 2B t + C = 0
    Real_v vx = direction.x() * ellipticalcone.invDx;
    Real_v vy = direction.y() * ellipticalcone.invDy;
    Real_v vz = direction.z();
    Real_v Ar = vx * vx + vy * vy;
    Real_v Br = px * vx + py * vy;
    Real_v Cr = px * px + py * py;
    // 1) Check if A = 0
    // If so, slightly modify vz to avoid degeneration of the quadratic equation
    // The magnitude of vz will be modified in a way that point is leaving the solid
    bool tinyA = vecCore::math::Abs(Ar - vz * vz) < kTwoEpsilon * vz * vz;
    if (tinyA) vz += vecCore::math::Abs(vz) * kTwoEpsilon;

    Real_v Az = vz * vz;
    Real_v Bz = pz0 * vz;
    Real_v Cz = pz0 * pz0;
    Real_v A  = Ar - Az;
    Real_v B  = Br - Bz;
    Real_v B0 = Br - pz0 * direction.z(); // B calculated with original v.z()
    Real_v C  = Cr - Cz;
    Real_v D  = B * B - A * C;
    if (sfr <= Real_v(0.) && D < Real_v(0.)) D = Real_v(0.);

    // 2) Check if scratching (D < eps & A > 0) or no intersection (D < 0)
    // 3) if (D < eps & A < 0) then trajectory traverses the apex area - continue calculation
    bool done = false;
    done |= (D < Real_v(0.)) || (D < kTwoEpsilon * B * B && A >= Real_v(0.));
    if (done) return;

    // Find intersection with Z planes
    Real_v tzout = kMaximum;
    if (vz != Real_v(0.)) tzout = (vecCore::math::CopySign(Real_v(ellipticalcone.fZCut), vz) - pz) / direction.z();

    // Find roots of the quadratic equation
    Real_v tmp = -B - vecCore::math::CopySign(vecCore::math::Sqrt(D), B);
    Real_v t1  = tmp / A;
    Real_v t2(0.);
    if (tmp != Real_v(0.)) t2 = C / tmp;
    if (tinyA && B0 != Real_v(0.)) t2 = -C / (Real_v(2.) * B0); // A ~ 0, t = -C / 2B
    Real_v tmin = vecCore::math::Min(t1, t2);
    Real_v tmax = vecCore::math::Max(t1, t2);

    // Set default - intersection with lower nappe (A > 0)
    Real_v trout = tmax;
    // Check if intersection with upper nappe only or flying away, return 0
    done |= ((A >= Real_v(0.) && pz0 + vz * tmax >= Real_v(0.)) || (pz0 >= Real_v(0.) && vz >= Real_v(0.)));
    if (done) return;

    // Check if intersection with both nappes (A < 0)
    if (A < Real_v(0.)) {
      trout = Real_v(kInfLength);
      if (vz > Real_v(0.)) trout = tmin;
    }

    // Set distance
    // No special check for inside points, distance for inside points will be negative
    distance = vecCore::math::Min(tzout, trout);
  }

  /// @brief Computes an outside safety estimate to the elliptical cone.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param[out] safety Maximum of the lateral and z-cut outside distances, clamped to zero in tolerance.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &ellipticalcone,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v px = point.x() * ellipticalcone.invDx;
    Real_v py = point.y() * ellipticalcone.invDy;
    Real_v pz = point.z();
    Real_v hp = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v ds = (hp - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    Real_v dz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    safety    = vecCore::math::Max(ds, dz);
    if (vecCore::math::Abs(safety) <= kHalfTolerance) safety = Real_v(0.);
  }

  /// @brief Computes an inside safety estimate to the elliptical cone boundary.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param[out] safety Minimum of the lateral and z-cut inside distances, clamped to zero in tolerance.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &ellipticalcone,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v px = point.x() * ellipticalcone.invDx;
    Real_v py = point.y() * ellipticalcone.invDy;
    Real_v pz = point.z();
    Real_v hp = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v ds = (ellipticalcone.fDz - hp) * ellipticalcone.cosAxisMin;
    Real_v dz = ellipticalcone.fZCut - vecCore::math::Abs(pz);
    safety    = vecCore::math::Min(ds, dz);
    if (vecCore::math::Abs(safety) <= kHalfTolerance) safety = Real_v(0.);
  }

  /// @brief Returns the outward normal for a point on the elliptical cone boundary.
  /// @tparam Real_v Scalar floating-point type used by the query.
  /// @param ellipticalcone Cached unplaced cone data.
  /// @param point Point in local coordinates.
  /// @param[out] valid Set to false when the point is not within surface tolerance.
  /// @return Unit normal on the closest boundary; edge normals average the adjacent surfaces.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, bool &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    //   In case if the point is further than kHalfTolerance from the surface, set valid=false
    //   Must return a valid vector (even if the point is not on the surface)
    //
    //   On an edge provide an average normal of the corresponding base and lateral surface
    Vector3D<Real_v> normal(0., 0., 0.);
    valid = true;

    // Check z planes
    Real_v px = point.x();
    Real_v py = point.y();
    Real_v pz = point.z();
    Real_v dz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    if (vecCore::math::Abs(dz) <= kHalfTolerance) normal[2] = vecCore::math::Sign(pz);

    // Check lateral surface
    Real_v nx = px * ellipticalcone.invDx * ellipticalcone.invDx;
    Real_v ny = py * ellipticalcone.invDy * ellipticalcone.invDy;
    Real_v nz = vecCore::math::Sqrt(px * nx + py * ny);
    if ((nx * nx + ny * ny) == Real_v(0.)) nz = Real_v(1.); // z-axis
    Vector3D<Real_v> nside(nx, ny, nz);
    Real_v ds = (nz + pz - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    if (vecCore::math::Abs(ds) <= kHalfTolerance) normal = (normal + nside.Unit()).Unit();

    // Check if done
    bool done = normal.Mag2() > Real_v(0.);
    if (done) return normal;

    // Point is not on the surface - normally, this should never be
    // Return normal to the nearest surface
    valid     = false;
    normal[2] = vecCore::math::Sign(pz);
    if (ds > dz) normal = nside.Unit();
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_
