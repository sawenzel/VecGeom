/// @file HalfSpaceImplementation.h
/// @brief Navigation kernel for the infinite half-space primitive.

#ifndef VECGEOM_VOLUMES_KERNEL_HALFSPACEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_HALFSPACEIMPLEMENTATION_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/HalfSpaceStruct.h"

#include <VecCore/VecCore>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct HalfSpaceImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, HalfSpaceImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedHalfSpace;
class UnplacedHalfSpace;

/// @brief Kernel implementation for a plane-bounded infinite half-space.
/// @details The stored normal is a unit vector pointing outside the material
/// side. Points with signed distance `(p - point).Dot(normal) <= 0` are in the
/// half-space, with the usual VecGeom tolerance band classified as surface.
struct HalfSpaceImplementation {
  using PlacedShape_t    = PlacedHalfSpace;
  using UnplacedStruct_t = HalfSpaceStruct<Precision>;
  using UnplacedVolume_t = UnplacedHalfSpace;

  /// @brief Compute signed distance to the limiting plane.
  /// @tparam Real_v Scalar floating-point type used by the kernel.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @return Positive distance on the outside side, negative on the material side.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v SignedDistance(UnplacedStruct_t const &halfspace,
                                                                            Vector3D<Real_v> const &point)
  {
    return (point - Vector3D<Real_v>(halfspace.fPoint)).Dot(Vector3D<Real_v>(halfspace.fNormal));
  }

  /// @brief Test whether a point is contained by the tolerated half-space.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param[out] inside True for material-side and surface-band points.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &halfspace,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    inside = SignedDistance(halfspace, point) <= Real_v(kTolerance);
  }

  /// @brief Classify a point with the half-space inside convention.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param[out] inside `kInside`, `kSurface`, or `kOutside`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &halfspace,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    const Real_v signedDistance = SignedDistance(halfspace, point);
    inside                      = EInside::kSurface;
    if (signedDistance < Real_v(-kTolerance)) inside = Inside_t(EInside::kInside);
    if (signedDistance > Real_v(kTolerance)) inside = Inside_t(EInside::kOutside);
  }

  /// @brief Compute distance from outside to enter the half-space.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param direction Unit ray direction in local coordinates.
  /// @param stepMax Maximum accepted step.
  /// @param[out] distance Entry distance, `kInfLength` for a miss, or `-1` when already inside.
  /// @details A surface point only enters at zero when the direction points
  /// toward the material side.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &halfspace,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    distance                    = kInfLength;
    const Real_v signedDistance = SignedDistance(halfspace, point);
    if (signedDistance < Real_v(-kTolerance)) {
      distance = Real_v(-1.);
      return;
    }

    const Real_v nDotDir = Vector3D<Real_v>(halfspace.fNormal).Dot(direction);
    if (signedDistance <= Real_v(kTolerance)) {
      if (nDotDir < Real_v(-kTolerance)) distance = Real_v(0.);
      return;
    }

    if (nDotDir >= Real_v(-kTolerance)) return;
    const Real_v candidate = -signedDistance / nDotDir;
    if (candidate >= Real_v(0.) && candidate < stepMax) distance = candidate;
  }

  /// @brief Compute distance from inside to leave the half-space.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param direction Unit ray direction in local coordinates.
  /// @param stepMax Maximum accepted step.
  /// @param[out] distance Exit distance, `kInfLength` for no plane crossing, or `-1` when already outside.
  /// @details A surface point only exits at zero when the direction points
  /// along the outward normal.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &halfspace,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    distance                    = kInfLength;
    const Real_v signedDistance = SignedDistance(halfspace, point);
    if (signedDistance > Real_v(kTolerance)) {
      distance = Real_v(-1.);
      return;
    }

    const Real_v nDotDir = Vector3D<Real_v>(halfspace.fNormal).Dot(direction);
    if (signedDistance >= Real_v(-kTolerance)) {
      if (nDotDir > Real_v(kTolerance)) distance = Real_v(0.);
      return;
    }

    if (nDotDir <= Real_v(kTolerance)) return;
    const Real_v candidate = -signedDistance / nDotDir;
    if (candidate >= Real_v(0.) && candidate < stepMax) distance = candidate;
  }

  /// @brief Compute safety from outside to the half-space boundary.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param[out] safety Positive outside distance, zero in the surface band, or `-1` when already inside.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &halfspace,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    const Real_v signedDistance = SignedDistance(halfspace, point);
    safety                      = signedDistance;
    if (signedDistance < Real_v(-kTolerance)) {
      safety = Real_v(-1.);
      return;
    }
    if (signedDistance <= Real_v(kTolerance)) safety = Real_v(0.);
  }

  /// @brief Compute safety from inside to the half-space boundary.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param[out] safety Positive material-side distance, zero in the surface band, or `-1` when outside.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &halfspace,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    const Real_v signedDistance = SignedDistance(halfspace, point);
    safety                      = -signedDistance;
    if (signedDistance > Real_v(kTolerance)) {
      safety = Real_v(-1.);
      return;
    }
    if (signedDistance >= Real_v(-kTolerance)) safety = Real_v(0.);
  }

  /// @brief Return the outward plane normal at a local query point.
  /// @param halfspace Plane data with normalized outward normal.
  /// @param point Query point in local coordinates.
  /// @param[out] valid True when @p point lies in the tolerated surface band.
  /// @return Unit outward plane normal.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &halfspace,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {
    using vecCore::math::Abs;
    valid = Abs(SignedDistance(halfspace, point)) <= Real_v(kTolerance);
    return Vector3D<Real_v>(halfspace.fNormal);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_HALFSPACEIMPLEMENTATION_H_
