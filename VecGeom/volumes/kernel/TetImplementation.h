// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file volumes/kernel/TetImplementation.h
/// @brief Tetrahedron kernel helpers and navigation entry points.
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/TetStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TetImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TetImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTet;
template <typename T>
struct TetStruct;
class UnplacedTet;

/// @brief Kernel implementation for tetrahedra.
/// @details A tetrahedron is represented as the intersection of four outward
/// face half-spaces, with signed plane distance `n.Dot(point) + d <= 0` inside.
/// Navigation methods use scalar four-plane predicates and clamp
/// tolerance-scale boundary crossings to the VecGeom zero-distance convention.
struct TetImplementation {

  using PlacedShape_t    = PlacedTet;
  using UnplacedStruct_t = TetStruct<Precision>;
  using UnplacedVolume_t = UnplacedTet;

  /// @brief Test whether a point is inside or on the tolerated tet boundary.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local point to test.
  /// @param[out] inside True unless the point is strictly outside a face.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &tet,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused = false, outside = false;
    GenericKernelForContainsAndInside<Real_v, false>(tet, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local point to classify.
  /// @param[out] inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &tet,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, true>(tet, point, completelyinside, completelyoutside);
    inside = completelyoutside ? EInside::kOutside : (completelyinside ? EInside::kInside : EInside::kSurface);
  }

  /// @brief Compute strict inside/outside flags from the maximum face distance.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam ForInside When true, also compute the strict-inside flag.
  /// @param tet Tetrahedron runtime data.
  /// @param localPoint Local point to classify.
  /// @param[out] completelyinside True when all face distances are below `-kHalfTolerance`.
  /// @param[out] completelyoutside True when any face distance is above `kHalfTolerance`.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &tet, Vector3D<Real_v> const &localPoint, bool &completelyinside, bool &completelyoutside)
  {
    const Real_v dist0  = Vector3D<Real_v>(tet.fPlane[0].n).Dot(localPoint) + tet.fPlane[0].d;
    const Real_v dist1  = Vector3D<Real_v>(tet.fPlane[1].n).Dot(localPoint) + tet.fPlane[1].d;
    const Real_v dist2  = Vector3D<Real_v>(tet.fPlane[2].n).Dot(localPoint) + tet.fPlane[2].d;
    const Real_v dist3  = Vector3D<Real_v>(tet.fPlane[3].n).Dot(localPoint) + tet.fPlane[3].d;
    const Real_v safety = Max(Max(Max(dist0, dist1), dist2), dist3);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  /// @brief Compute distance from an exterior point to enter the tetrahedron.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Entry distance, or `kInfLength` when no valid entry exists.
  /// @details Uses a convex four-plane slab interval. Outside faces must be
  /// crossed inward; the entry distance is the maximum inward crossing and the
  /// continuation limit is the minimum outward crossing. Accepted
  /// tolerance-scale entry crossings are reported as zero.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &tet,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    distance                        = -kInfLength;
    Real_v distanceOut              = kInfLength;
    const Real_v distanceTolerance  = kToleranceDist<Real_v>;
    const Real_v directionTolerance = kToleranceStrict<Real_v>;
    bool missed                     = false;
    for (int i = 0; i < 4; ++i) {
      const Real_v proj = Vector3D<Real_v>(tet.fPlane[i].n).Dot(direction);
      const Real_v safe = Vector3D<Real_v>(tet.fPlane[i].n).Dot(point) + tet.fPlane[i].d;
      const Real_v dist = -safe / NonZero(proj);
      if (safe > distanceTolerance && proj >= -directionTolerance) missed = true;
      Real_v crossing = dist;
      if (Abs(crossing) <= distanceTolerance || (Abs(safe) <= distanceTolerance && Abs(proj) <= directionTolerance)) {
        crossing = Real_v(0.);
      }
      if (proj < Real_v(0.)) distance = Max(distance, crossing);
      if (safe <= distanceTolerance && proj > Real_v(0.)) distanceOut = Min(distanceOut, crossing);
    }

    if (missed || distance >= distanceOut || distanceOut <= distanceTolerance) {
      distance = Real_v(kInfLength);
    } else if (Abs(distance) <= distanceTolerance) {
      distance = Real_v(0.);
    }
  }

  /// @brief Compute distance from an interior point to leave the tetrahedron.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Exit distance, or `-1` when the point is outside beyond tolerance.
  /// @details The exit is the nearest forward crossing of an outward-facing
  /// plane. Boundary crossings with tolerance-scale projected distance are
  /// reported as zero, matching the surface navigation convention.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &tet,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    distance      = kInfLength;
    Real_v safety = -kInfLength;

    const Real_v distanceTolerance  = kToleranceDist<Real_v>;
    const Real_v directionTolerance = kToleranceStrict<Real_v>;
    for (int i = 0; i < 4; ++i) {
      const Real_v proj = Vector3D<Real_v>(tet.fPlane[i].n).Dot(direction);
      const Real_v safe = Vector3D<Real_v>(tet.fPlane[i].n).Dot(point) + tet.fPlane[i].d;
      safety            = Max(safety, safe);
      Real_v crossing   = -safe / NonZero(proj);
      if (Abs(crossing) <= distanceTolerance || (Abs(safe) <= distanceTolerance && Abs(proj) <= directionTolerance)) {
        crossing = Real_v(0.);
      }
      if (proj > Real_v(0.)) distance = Min(distance, crossing);
    }

    if (safety > distanceTolerance) {
      distance = Real_v(-1.);
    } else if (Abs(distance) <= distanceTolerance) {
      distance = Real_v(0.);
    }
  }

  /// @brief Compute safety from an outside point to the tetrahedron.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local point.
  /// @param[out] safety Maximum signed face distance, clamped to zero in the surface band.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &tet,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    const Real_v dist0 = point.Dot(tet.fPlane[0].n) + tet.fPlane[0].d;
    const Real_v dist1 = point.Dot(tet.fPlane[1].n) + tet.fPlane[1].d;
    const Real_v dist2 = point.Dot(tet.fPlane[2].n) + tet.fPlane[2].d;
    const Real_v dist3 = point.Dot(tet.fPlane[3].n) + tet.fPlane[3].d;
    safety             = Max(Max(Max(dist0, dist1), dist2), dist3);
    safety             = Abs(safety) <= kHalfTolerance ? Real_v(0.) : safety;
  }

  /// @brief Compute safety from an inside point to leave the tetrahedron.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local point.
  /// @param[out] safety Negative maximum signed face distance, clamped to zero in the surface band.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &tet,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    const Real_v dist0 = point.Dot(tet.fPlane[0].n) + tet.fPlane[0].d;
    const Real_v dist1 = point.Dot(tet.fPlane[1].n) + tet.fPlane[1].d;
    const Real_v dist2 = point.Dot(tet.fPlane[2].n) + tet.fPlane[2].d;
    const Real_v dist3 = point.Dot(tet.fPlane[3].n) + tet.fPlane[3].d;
    safety             = -Max(Max(Max(dist0, dist1), dist2), dist3);
    safety             = Abs(safety) <= kHalfTolerance ? Real_v(0.) : safety;
  }

  /// @brief Compute an outward surface normal for the tetrahedron.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tet Tetrahedron runtime data.
  /// @param point Local point expected on or near the surface.
  /// @param[out] valid True when the point is within tolerance of at least one face.
  /// @return Unit outward normal. Edge/corner normals are normalized sums of
  /// adjacent face normals; invalid input returns the nearest face normal.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v dist[4];
    Vector3D<Real_v> n = tet.fPlane[0].n;
    dist[0]            = n.Dot(point) + tet.fPlane[0].d;
    if (Abs(dist[0]) <= kHalfTolerance) normal = normal + tet.fPlane[0].n;
    n       = tet.fPlane[1].n;
    dist[1] = n.Dot(point) + tet.fPlane[1].d;
    if (Abs(dist[1]) <= kHalfTolerance) normal = normal + tet.fPlane[1].n;
    n       = tet.fPlane[2].n;
    dist[2] = n.Dot(point) + tet.fPlane[2].d;
    if (Abs(dist[2]) <= kHalfTolerance) normal = normal + tet.fPlane[2].n;
    n       = tet.fPlane[3].n;
    dist[3] = n.Dot(point) + tet.fPlane[3].d;
    if (Abs(dist[3]) <= kHalfTolerance) normal = normal + tet.fPlane[3].n;
    if (normal.Mag2() > Real_v(1.)) normal = normal.Unit();

    bool done = normal.Mag2() > Real_v(0.);
    if (done) return normal;

    // Point is not on the surface - normally, this should never be.
    // Return normal of the nearest face.
    //
    valid = false;

    Real_v safety(-kInfLength);
    for (int i = 0; i < 4; ++i) {
      if (dist[i] > safety) {
        normal = tet.fPlane[i].n;
        safety = dist[i];
      }
    }
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_
