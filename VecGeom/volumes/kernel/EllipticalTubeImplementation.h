// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file EllipticalTubeImplementation.h
/// @brief Navigation kernels for the elliptical tube solid.
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPTICALTUBEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPTICALTUBEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipticalTubeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipticalTubeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipticalTubeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipticalTube;
template <typename T>
struct EllipticalTubeStruct;
class UnplacedEllipticalTube;

/// @brief Implements scalar navigation kernels for `UnplacedEllipticalTube`.
///
/// @details
/// The implementation maps the elliptical cross-section to a circular cylinder
/// with cached x/y scale factors, then combines the lateral-cylinder interval
/// with the z-slab interval for distance queries.
struct EllipticalTubeImplementation {

  using PlacedShape_t    = PlacedEllipticalTube;
  using UnplacedStruct_t = EllipticalTubeStruct<Precision>;
  using UnplacedVolume_t = UnplacedEllipticalTube;

  /// @brief Test whether a local point is contained in or on the elliptical tube.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Bool_t Boolean-like output type.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local point to test.
  /// @param inside Set to true unless @p point is outside the tolerated surface.
  template <typename Real_v, typename Bool_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &ellipticaltube,
                                                                    Vector3D<Real_v> const &point, Bool_t &inside)
  {
    bool unused = false, outside = false;
    GenericKernelForContainsAndInside<Real_v, bool, false>(ellipticaltube, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Inside_t Integer-like type used for `EInside` values.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local point to classify.
  /// @param inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &ellipticaltube,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside = false, completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, bool, true>(ellipticaltube, point, completelyinside, completelyoutside);
    inside = Inside_t(EInside::kSurface);
    if (completelyoutside) inside = Inside_t(EInside::kOutside);
    if (completelyinside) inside = Inside_t(EInside::kInside);
  }

  /// @brief Compute strict inside/outside flags for point classification.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Bool_t Boolean-like output type.
  /// @tparam ForInside When true, also compute the strict-inside flag.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local point to classify.
  /// @param completelyinside Set when @p point is separated from all surfaces by the inside tolerance.
  /// @param completelyoutside Set when @p point is outside the tolerated surface.
  ///
  /// @details
  /// The radial part is evaluated after scaling the cross-section to a circle;
  /// the maximum of radial excess and z-slab excess determines classification.
  template <typename Real_v, typename Bool_t, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &ellipticaltube, Vector3D<Real_v> const &point, Bool_t &completelyinside,
      Bool_t &completelyoutside)
  {
    Real_v x      = point.x() * ellipticaltube.fSx;
    Real_v y      = point.y() * ellipticaltube.fSy;
    Real_v distR  = ellipticaltube.fQ1 * (x * x + y * y) - ellipticaltube.fQ2;
    Real_v distZ  = vecCore::math::Abs(point.z()) - ellipticaltube.fDz;
    Real_v safety = vecCore::math::Max(distR, distZ);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  /// @brief Compute the distance from outside the elliptical tube to first entry.
  /// @tparam Real_v Floating-point scalar type.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param distance Set to the entry distance or `kInfLength` when there is no valid entry.
  ///
  /// @details
  /// Far starts moving toward the solid are shifted closer to the bounding
  /// sphere before solving. The radial equation is solved in scaled cylinder
  /// coordinates and intersected with the z-slab interval. Near-tangent
  /// lateral candidates are rejected with the cached scratch threshold.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &ellipticaltube,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    distance = kInfLength;
    Real_v offset(0.);
    Vector3D<Real_v> pcur(point);

    // Move point closer, if required
    Real_v Rfar2(1024. * ellipticaltube.fRsph * ellipticaltube.fRsph); // 1024 = 32 * 32
    if ((pcur.Mag2() > Rfar2) && (direction.Dot(point) < Real_v(0.))) {
      offset = pcur.Mag() - Real_v(2.) * ellipticaltube.fRsph;
      pcur += offset * direction;
    }

    // Scale elliptical tube to cylinder
    Real_v px = pcur.x() * ellipticaltube.fSx;
    Real_v py = pcur.y() * ellipticaltube.fSy;
    Real_v pz = pcur.z();
    Real_v vx = direction.x() * ellipticaltube.fSx;
    Real_v vy = direction.y() * ellipticaltube.fSy;
    Real_v vz = direction.z();

    // Find intersection with Z planes
    Real_v invz  = Real_v(-1.) / NonZero(vz);
    Real_v dz    = vecCore::math::CopySign(Real_v(ellipticaltube.fDz), invz);
    Real_v tzmin = (pz - dz) * invz;
    Real_v tzmax = (pz + dz) * invz;

    // Find intersection with lateral surface, solve equation: A t^2 + 2B t + C = 0
    Real_v rr = px * px + py * py;
    Real_v A  = vx * vx + vy * vy;
    Real_v B  = px * vx + py * vy;
    Real_v C  = rr - ellipticaltube.fR * ellipticaltube.fR;
    Real_v D  = B * B - A * C;

    // Check if point leaving shape
    Real_v distZ     = vecCore::math::Abs(pz) - ellipticaltube.fDz;
    Real_v distR     = ellipticaltube.fQ1 * rr - ellipticaltube.fQ2;
    bool parallelToZ = (A < kEpsilon || vecCore::math::Abs(vz) >= Real_v(1.));
    bool leaving     = (distZ >= -kHalfTolerance && pz * vz >= Real_v(0.)) ||
                       (distR >= -kHalfTolerance && (B >= Real_v(0.) || parallelToZ));

    // Two special cases where D <= 0:
    //   1) trajectory parallel to Z axis (A = 0, B = 0, C - any, D = 0)
    //   2) touch (D = 0) or no intersection (D < 0) with lateral surface
    if (!leaving && parallelToZ) distance = tzmin + offset;                       // 1)
    bool done = (leaving || parallelToZ || D <= A * A * ellipticaltube.fScratch); // 2)
    if (done) return;

    // if (D <= A * A * ellipticaltube.fScratch) std::cerr << "=== SCRATCH D = " << D << std::endl;

    // Find roots of the quadratic
    Real_v tmp   = -B - vecCore::math::CopySign(vecCore::math::Sqrt(D), B);
    Real_v t1    = tmp / A;
    Real_v t2    = C / tmp;
    Real_v trmin = vecCore::math::Min(t1, t2);
    Real_v trmax = vecCore::math::Max(t1, t2);

    // Set distance
    // No special check for inside points, for inside points distance will be negative
    Real_v tin  = vecCore::math::Max(tzmin, trmin);
    Real_v tout = vecCore::math::Min(tzmax, trmax);
    if ((tout - tin) >= kHalfTolerance) distance = tin + offset;
  }

  /// @brief Compute the distance from inside the elliptical tube to first exit.
  /// @tparam Real_v Floating-point scalar type.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param distance Set to the exit distance, or `-1` when @p point is clearly outside.
  ///
  /// @details
  /// The method intersects the forward z-slab exit with the forward root of
  /// the scaled lateral-cylinder quadratic. Tangential or non-intersecting
  /// lateral candidates leave the cap distance as the selected exit.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &ellipticaltube,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    // Scale elliptical tube to cylinder
    Real_v px = point.x() * ellipticaltube.fSx;
    Real_v py = point.y() * ellipticaltube.fSy;
    Real_v pz = point.z();
    Real_v vx = direction.x() * ellipticaltube.fSx;
    Real_v vy = direction.y() * ellipticaltube.fSy;
    Real_v vz = direction.z();

    // Check if point is outside ("wrong side")
    Real_v rr    = px * px + py * py;
    Real_v distR = ellipticaltube.fQ1 * rr - ellipticaltube.fQ2;
    Real_v distZ = vecCore::math::Abs(pz) - ellipticaltube.fDz;
    bool outside = vecCore::math::Max(distR, distZ) > kHalfTolerance;
    distance     = Real_v(0.);
    if (outside) {
      distance = Real_v(-1.);
      return;
    }

    // Find intersection with Z planes
    Real_v tzmax = kMaximum;
    if (vz != Real_v(0.)) tzmax = (vecCore::math::CopySign(Real_v(ellipticaltube.fDz), vz) - pz) / vz;

    // Find intersection with lateral surface, solve equation: A t^2 + 2B t + C = 0
    Real_v A = vx * vx + vy * vy;
    Real_v B = px * vx + py * vy;
    Real_v C = rr - ellipticaltube.fR * ellipticaltube.fR;
    Real_v D = B * B - A * C;

    // Two cases where D <= 0:
    //   1) trajectory parallel to Z axis (A = 0, B = 0, C - any, D = 0)
    //   2) touch (D = 0) or no intersection (D < 0) with lateral surface
    bool parallelToZ = (A < kEpsilon || vecCore::math::Abs(vz) >= Real_v(1.));
    if (parallelToZ) {
      distance = tzmax; // 1)
      return;
    }
    bool done = (D <= Real_v(0.)); // 2)
    if (done) return;

    // Set distance
    Real_v sqrtD = vecCore::math::Sqrt(D);
    if (B >= Real_v(0.)) {
      distance = vecCore::math::Min(tzmax, -C / (sqrtD + B));
    } else {
      distance = vecCore::math::Min(tzmax, (sqrtD - B) / A);
    }
  }

  /// @brief Compute safety from an outside point to the elliptical tube.
  /// @tparam Real_v Floating-point scalar type.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local point.
  /// @param safety Set to the maximum of radial and z-slab safety, clamped to zero in the surface band.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &ellipticaltube,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v x     = point.x() * ellipticaltube.fSx;
    Real_v y     = point.y() * ellipticaltube.fSy;
    Real_v distR = vecCore::math::Sqrt(x * x + y * y) - ellipticaltube.fR;
    Real_v distZ = vecCore::math::Abs(point.z()) - ellipticaltube.fDz;

    safety = vecCore::math::Max(distR, distZ);
    if (vecCore::math::Abs(safety) <= kHalfTolerance) safety = Real_v(0.);
  }

  /// @brief Compute safety from an inside point to leave the elliptical tube.
  /// @tparam Real_v Floating-point scalar type.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local point.
  /// @param safety Set to the smaller remaining distance to the lateral surface or z cap.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &ellipticaltube,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v x     = point.x() * ellipticaltube.fSx;
    Real_v y     = point.y() * ellipticaltube.fSy;
    Real_v distR = ellipticaltube.fR - vecCore::math::Sqrt(x * x + y * y);
    Real_v distZ = ellipticaltube.fDz - vecCore::math::Abs(point.z());

    safety = vecCore::math::Min(distR, distZ);
    if (vecCore::math::Abs(safety) <= kHalfTolerance) safety = Real_v(0.);
  }

  /// @brief Compute an outward surface normal.
  /// @tparam Real_v Floating-point scalar type.
  /// @param ellipticaltube Cached elliptical tube data.
  /// @param point Local point.
  /// @param valid Set when @p point is in the tolerated surface band.
  /// @return Unit normal on the lateral surface, cap, or their averaged edge normal.
  ///
  /// @details
  /// Lateral normals use the gradient of the unscaled ellipse. If the point is
  /// not on a tolerated surface, a nearest-surface fallback is returned with
  /// @p valid set to false.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &ellipticaltube, Vector3D<Real_v> const &point, bool &valid)
  {
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v x     = point.x() * ellipticaltube.fSx;
    Real_v y     = point.y() * ellipticaltube.fSy;
    Real_v distR = ellipticaltube.fQ1 * (x * x + y * y) - ellipticaltube.fQ2;
    if (vecCore::math::Abs(distR) <= kHalfTolerance) {
      normal = Vector3D<Real_v>(point.x() * ellipticaltube.fDDy, point.y() * ellipticaltube.fDDx, 0.).Unit();
    }

    Real_v distZ = vecCore::math::Abs(point.z()) - ellipticaltube.fDz;
    if (vecCore::math::Abs(distZ) <= kHalfTolerance) normal[2] = vecCore::math::Sign(point[2]);
    if (normal.Mag2() > Real_v(1.)) normal = normal.Unit();

    bool done = normal.Mag2() > Real_v(0.);
    if (done) return normal;

    // Point is not on the surface - normally, this should never be
    // Return normal to the nearest surface
    valid     = false;
    normal[2] = vecCore::math::Sign(point[2]);
    distR     = vecCore::math::Sqrt(x * x + y * y) - ellipticaltube.fR;
    if (distR > distZ && (x * x + y * y) > Real_v(0.)) {
      normal = Vector3D<Real_v>(point.x() * ellipticaltube.fDDy, point.y() * ellipticaltube.fDDx, 0.).Unit();
    }
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPTICALTUBEIMPLEMENTATION_H_
