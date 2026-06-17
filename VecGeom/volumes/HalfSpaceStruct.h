/// @file HalfSpaceStruct.h
/// @brief Runtime data for the plane-bounded half-space primitive.

#ifndef VECGEOM_VOLUMES_HALFSPACESTRUCT_H_
#define VECGEOM_VOLUMES_HALFSPACESTRUCT_H_

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// @brief Plane point and normalized outward normal defining a half-space.
/// @details The material side satisfies `(p - fPoint).Dot(fNormal) <= 0`.
template <typename T = double>
struct HalfSpaceStruct {
  Vector3D<T> fPoint;  ///< Point on the limiting plane.
  Vector3D<T> fNormal; ///< Unit normal pointing outside the half-space.

  /// @brief Construct the default half-space `z <= 0`.
  VECCORE_ATT_HOST_DEVICE
  HalfSpaceStruct() : fPoint(0.), fNormal(0., 0., 1.) {}

  /// @brief Construct a half-space from one plane point and an outward normal.
  /// @param point Point lying on the limiting plane.
  /// @param normal Non-zero normal; it is normalized before storage.
  /// @details The normal points to the outside of the half-space, so material is
  /// on the side where `(p - point).Dot(normal) <= 0`.
  VECCORE_ATT_HOST_DEVICE
  HalfSpaceStruct(Vector3D<T> const &point, Vector3D<T> const &normal) : fPoint(point), fNormal(normal)
  {
    VECGEOM_VALIDATE(fNormal.Mag2() > T(0.), << "Half-space normal must be non-zero");
    fNormal.Normalize();
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_HALFSPACESTRUCT_H_
