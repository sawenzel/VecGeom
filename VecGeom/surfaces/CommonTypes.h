#ifndef VECGEOM_SURFACE_COMMONTYPES_H
#define VECGEOM_SURFACE_COMMONTYPES_H

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/base/Vector2D.h>
#include <VecGeom/base/Vector3D.h>

namespace vgbrep {

///< VecGeom type aliases
template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

template <typename Real_t>
using Vector2D = vecgeom::Vector2D<Real_t>;

using Transformation = vecgeom::Transformation3D;

///< Supported surface types
enum SurfaceType { kPlanar, kCylindrical, kConical, kSpherical, kTorus, kGenSecondOrder };

///< Supported frame types
///> kRangeZ      <- range along z-axis
///> kRing        <- a "ring" range on a plane
///> kZPhi        <- z and phi range on a cylinder
///> kRangeSph    <- theta and phi range on a sphere
///> kWindow      <- rectangular range in xy-plane
///> kTriangle    <- triangular range in xy-plane
enum FrameType { kRangeZ, kRing, kZPhi, kRangeSph, kWindow, kTriangle, kQuadrilateral };

// Aliases for different usages of Vec2D.
template <typename Real_t>
using Range = Vector2D<Real_t>;

template <typename Real_t>
using AngleVector = Vector2D<Real_t>;

template <typename Real_t>
using Point2D = Vector2D<Real_t>;

/// @brief Data for cylindrical and spherical surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t, typename Real_s = Real_t>
struct CylData {
  Real_t radius{0}; ///< Cylinder radius. Stored negative if flipped.

  CylData() = default;
  CylData(Real_s rad, bool flip = false) : radius(flip ? -rad : rad) {}

  VECCORE_ATT_HOST_DEVICE
  Real_s Radius() const { return std::abs(Real_s(radius)); }
  VECCORE_ATT_HOST_DEVICE
  bool IsFlipped() const { return radius < 0; }
};

template <typename Real_t, typename Real_s = Real_t>
using SphData = CylData<Real_t, Real_s>;

/// @brief Data for conical surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t, typename Real_s = Real_t>
struct ConeData {
  Real_t radius{0}; ///< Cone radus at Z = 0. Stored negative if flipped.
  Real_t slope{0};  ///< Cone slope  --> for cyl extension this would be 0

  ConeData() = default;
  ConeData(Real_s rad, Real_s slope, bool flip = false) : radius(flip ? -rad : rad), slope(slope) {}
  VECCORE_ATT_HOST_DEVICE
  Real_s Radius() const { return std::abs(Real_s(radius)); }
  VECCORE_ATT_HOST_DEVICE
  Real_s RadiusZ(Real_s z) const { return Radius() + z * slope; }
  VECCORE_ATT_HOST_DEVICE
  Real_s Slope() const { return slope; }
  VECCORE_ATT_HOST_DEVICE
  bool IsFlipped() const { return radius < 0; }
};

} // namespace vgbrep

#endif
