// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file volumes/kernel/EllipsoidImplementation.h
/// @brief Navigation kernels for the ellipsoid solid.
/// @author Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPSOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPSOIDIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipsoidStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>
#include <iomanip>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipsoidImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipsoidImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipsoid;
template <typename T>
struct EllipsoidStruct;
class UnplacedEllipsoid;

/// @brief Implements Ellipsoid classification, distance, safety, and normal kernels.
///
/// @details The implementation maps the ellipsoid to a sphere using cached scale
/// factors from `EllipsoidStruct`. Z cuts are handled in the scaled frame for
/// classification and distance root filtering, while safety values are converted
/// back to conservative local-space bounds where needed.
struct EllipsoidImplementation {

  using PlacedShape_t    = PlacedEllipsoid;
  using UnplacedStruct_t = EllipsoidStruct<Precision>;
  using UnplacedVolume_t = UnplacedEllipsoid;

  /// @brief Tests whether a point is inside or on the ellipsoid.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param inside Output flag set to true unless the point is outside.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &ellipsoid,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    bool unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, false>(ellipsoid, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classifies a point against the ellipsoid and its z cuts.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param inside Output classification as `EInside`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &ellipsoid,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, true>(ellipsoid, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = Inside_t(EInside::kOutside);
    if (completelyinside) inside = Inside_t(EInside::kInside);
  }

  /// @brief Computes the shared inside/outside predicates.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param completelyinside Output true when the point is strictly inside all bounds.
  /// @param completelyoutside Output true when the point is outside at least one bound.
  /// @tparam ForInside Whether the strict-inside predicate is needed.
  ///
  /// @details The point is scaled to the auxiliary sphere. The radial and z-cut
  /// distances are combined into a single tolerance-band predicate.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &ellipsoid, Vector3D<Real_v> const &point, bool &completelyinside, bool &completelyoutside)
  {
    Real_v x      = point.x() * ellipsoid.fSx;
    Real_v y      = point.y() * ellipsoid.fSy;
    Real_v z      = point.z() * ellipsoid.fSz;
    Real_v distZ  = vecCore::math::Abs(z - ellipsoid.fScZMidCut) - ellipsoid.fScZDimCut;
    Real_v distR  = ellipsoid.fQ1 * (x * x + y * y + z * z) - ellipsoid.fQ2;
    Real_v safety = vecCore::math::Max(distZ, distR);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  /// @brief Computes the distance from an outside point to the ellipsoid.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param direction Unit direction in the local solid frame.
  /// @param distance Output distance, or `kInfLength` if the ray misses.
  ///
  /// @details Far-away points are translated closer along the ray before scaling
  /// to the auxiliary sphere. Candidate hits are the overlap of the sphere
  /// interval and the z-cut interval.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &ellipsoid,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    distance = kInfLength;
    Real_v offset(0.);
    Vector3D<Real_v> pcur(point);

    // Move point closer, if required
    Real_v Rfar2(1024. * ellipsoid.fRsph * ellipsoid.fRsph); // 1024 = 32 * 32
    if ((pcur.Mag2() > Rfar2) && (direction.Dot(point) < Real_v(0.))) {
      offset = pcur.Mag() - Real_v(2.) * ellipsoid.fRsph;
      pcur += offset * direction;
    }

    // Scale ellipsoid to sphere
    Real_v px = pcur.x() * ellipsoid.fSx;
    Real_v py = pcur.y() * ellipsoid.fSy;
    Real_v pz = pcur.z() * ellipsoid.fSz;
    Real_v vx = direction.x() * ellipsoid.fSx;
    Real_v vy = direction.y() * ellipsoid.fSy;
    Real_v vz = direction.z() * ellipsoid.fSz;

    // Check if point is leaving the solid
    Real_v pzcut = pz - ellipsoid.fScZMidCut;
    Real_v dzcut = Real_v(ellipsoid.fScZDimCut);
    Real_v distZ = vecCore::math::Abs(pzcut) - dzcut;

    Real_v rr    = px * px + py * py + pz * pz;
    Real_v vv    = vx * vx + vy * vy + vz * vz;
    Real_v pv    = px * vx + py * vy + pz * vz;
    Real_v distR = ellipsoid.fQ1 * rr - ellipsoid.fQ2;
    bool leaving =
        (distZ >= -kHalfTolerance && pzcut * vz >= Real_v(0.)) || (distR >= -kHalfTolerance && pv >= Real_v(0.));
    if (leaving) return;

    // Find intersection with Z planes
    Real_v invz  = Real_v(-1.) / NonZero(vz);
    Real_v dz    = vecCore::math::CopySign(dzcut, invz);
    Real_v tzmin = (pzcut - dz) * invz;
    Real_v tzmax = (pzcut + dz) * invz;

    // Find intersection with sphere
    Real_v A   = vv;
    Real_v B   = pv;
    Real_v C   = (rr - ellipsoid.fR * ellipsoid.fR);
    Real_v D   = B * B - A * C;
    Real_v EPS = Real_v(2.) * rr * vv * kEpsilon;
    if (D <= EPS) {
      if (C <= Real_v(0.) && distR > -kTolerance && distR < kTolerance && B < -Real_v(0.5 * kTolerance) * A)
        distance = offset;
      return;
    }
    Real_v sqrtD = vecCore::math::Sqrt(D);
    Real_v trmin = (-B - sqrtD) / A;
    Real_v trmax = (-B + sqrtD) / A;

    // Set preliminary distances to in/out
    Real_v tmin = vecCore::math::Max(tzmin, trmin);
    Real_v tmax = vecCore::math::Min(tzmax, trmax);

    // Check if no intersection
    bool done = (tmax - tmin) <= kTolerance;

    // Set distance
    if (!done) distance = tmin + offset;
  }

  /// @brief Computes the distance from an inside point to the next boundary.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param direction Unit direction in the local solid frame.
  /// @param distance Output distance, zero for unresolved tangent/no-hit cases, or -1 for wrong-side input.
  ///
  /// @details The first exit is the smaller positive hit between the scaled
  /// sphere and the active z-cut plane.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &ellipsoid,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    // Scale ellipsoid to sphere
    Real_v px = point.x() * ellipsoid.fSx;
    Real_v py = point.y() * ellipsoid.fSy;
    Real_v pz = point.z() * ellipsoid.fSz;
    Real_v vx = direction.x() * ellipsoid.fSx;
    Real_v vy = direction.y() * ellipsoid.fSy;
    Real_v vz = direction.z() * ellipsoid.fSz;

    // Check if point is outside ("wrong side")
    Real_v pzcut = pz - ellipsoid.fScZMidCut;
    Real_v dzcut = Real_v(ellipsoid.fScZDimCut);
    Real_v distZ = vecCore::math::Abs(pzcut) - dzcut;

    Real_v rr    = px * px + py * py + pz * pz;
    Real_v vv    = vx * vx + vy * vy + vz * vz;
    Real_v pv    = px * vx + py * vy + pz * vz;
    Real_v distR = ellipsoid.fQ1 * rr - ellipsoid.fQ2;
    bool outside = vecCore::math::Max(distR, distZ) > kHalfTolerance;

    distance = Real_v(0.);
    if (outside) {
      distance = Real_v(-1.);
      return;
    }

    // Find intersection with Z planes
    Real_v tzmax = kMaximum;
    if (vz != Real_v(0.)) tzmax = (vecCore::math::CopySign(Real_v(dzcut), vz) - pzcut) / vz;

    // Find intersection with sphere
    Real_v B   = pv / vv;
    Real_v C   = (rr - ellipsoid.fR * ellipsoid.fR) / vv;
    Real_v D   = B * B - C;
    Real_v EPS = Real_v(2.) * rr * vv * kEpsilon;
    if (D <= EPS) {
      if (C <= Real_v(0.) && distR > -kTolerance && distR < kTolerance && B < -Real_v(0.5 * kTolerance))
        distance = vecCore::math::Min(tzmax, -Real_v(2.) * B);
      return;
    }
    Real_v sqrtD = vecCore::math::Sqrt(D);
    Real_v trmax = -B + sqrtD;

    // Set distance
    distance = vecCore::math::Min(tzmax, trmax);
  }

  /// @brief Computes a conservative safety from an outside point to the ellipsoid.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param safety Output lower bound to the solid boundary.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &ellipsoid,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v x = point.x() * ellipsoid.fSx;
    Real_v y = point.y() * ellipsoid.fSy;
    Real_v z = point.z() * ellipsoid.fSz;
    Real_v r = vecCore::math::Sqrt(x * x + y * y + z * z);
    // Set safety to zero if point is on surface
    Real_v safeZ = vecCore::math::Abs(z - ellipsoid.fScZMidCut) - ellipsoid.fScZDimCut;
    Real_v safeR = r - ellipsoid.fR;
    safety       = vecCore::math::Max(safeZ, safeR);
    if (vecCore::math::Abs(safety) <= kHalfTolerance) safety = Real_v(0.);
    // Adjust safety using bounding box
    Real_v distZ  = vecCore::math::Max(point.z() - ellipsoid.fZTopCut, ellipsoid.fZBottomCut - point.z());
    Real_v distXY = vecCore::math::Max(vecCore::math::Abs(point.x()) - ellipsoid.fXmax,
                                       vecCore::math::Abs(point.y()) - ellipsoid.fYmax);
    if (safety > Real_v(0.)) safety = vecCore::math::Max(safety, distZ, distXY);
  }

  /// @brief Computes a conservative safety from an inside point to the boundary.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param safety Output lower bound to the exit boundary.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &ellipsoid,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v x = point.x() * ellipsoid.fSx;
    Real_v y = point.y() * ellipsoid.fSy;
    Real_v z = point.z() * ellipsoid.fSz;
    // Set safety to zero if point is on surface
    Real_v safeR = ellipsoid.fR - vecCore::math::Sqrt(x * x + y * y + z * z);
    Real_v safeZ = ellipsoid.fScZDimCut - vecCore::math::Abs(z - ellipsoid.fScZMidCut);
    safety       = vecCore::math::Min(safeZ, safeR);
    if (vecCore::math::Abs(safety) <= kHalfTolerance) safety = Real_v(0.);
    // Adjust safety in z direction
    Real_v distZ = vecCore::math::Min(ellipsoid.fZTopCut - point.z(), point.z() - ellipsoid.fZBottomCut);
    if (safety > Real_v(0.)) safety = vecCore::math::Min(safeR, distZ);
  }

  /// @brief Computes the outward normal on an ellipsoid boundary point.
  /// @param ellipsoid Cached unplaced ellipsoid data.
  /// @param point Point in the local solid frame.
  /// @param valid Output false when the point is not within tolerance of a boundary.
  /// @return Unit normal for boundary points, or a nearest-boundary fallback normal.
  ///
  /// @details Side normals use the ellipsoid gradient. Points on a z-cut edge
  /// receive the normalized average of side and cut normals.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &ellipsoid,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v px   = point.x();
    Real_v py   = point.y();
    Real_v pz   = point.z();
    Real_v A    = ellipsoid.fDx;
    Real_v B    = ellipsoid.fDy;
    Real_v C    = ellipsoid.fDz;
    Real_v x    = px * ellipsoid.fSx;
    Real_v y    = py * ellipsoid.fSy;
    Real_v z    = pz * ellipsoid.fSz;
    Real_v mag2 = x * x + y * y + z * z;

    // Check lateral surface
    Real_v distR = ellipsoid.fQ1 * mag2 - ellipsoid.fQ2;
    if (vecCore::math::Abs(distR) <= kHalfTolerance) {
      normal = Vector3D<Real_v>(px / (A * A), py / (B * B), pz / (C * C)).Unit();
    }

    // Check z cuts
    Real_v distZ = vecCore::math::Abs(z - ellipsoid.fScZMidCut) - ellipsoid.fScZDimCut;
    if (vecCore::math::Abs(distZ) <= kHalfTolerance) normal[2] += vecCore::math::Sign(z - ellipsoid.fScZMidCut);

    // Average normal, if required
    Real_v normalMag2 = normal.Mag2();
    if (normalMag2 > 1.) {
      normal     = normal.Unit();
      normalMag2 = Real_v(1.);
    }
    if (normalMag2 > Real_v(0.)) return normal;

    // Point is not on the surface - normally, this should never be
    // Return normal to the nearest surface
    valid     = false;
    normal[2] = vecCore::math::Sign(z - ellipsoid.fScZMidCut);
    distR     = vecCore::math::Sqrt(mag2) - ellipsoid.fR;
    if (distR > distZ && mag2 > Real_v(0.)) {
      normal = Vector3D<Real_v>(px / (A * A), py / (B * B), pz / (C * C)).Unit();
    }
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPSOIDIMPLEMENTATION_H_
