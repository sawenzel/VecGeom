#ifndef VECGEOM_SURFACE_COMMONTYPES_H
#define VECGEOM_SURFACE_COMMONTYPES_H

#include <cassert>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/base/Vector2D.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/volumes/kernel/GenericKernels.h>

namespace vgbrep {

///< VecGeom type aliases
template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

template <typename Real_t>
using Vector2D       = vecgeom::Vector2D<Real_t>;
using Transformation = vecgeom::Transformation3D;

using logic_int = int;

/// @brief Operators used inside Boolean expressions
enum OperatorToken : logic_int {
  lbegin = logic_int(0x7FFFFFFF - 8),
  lfalse = lbegin, ///< Push 'false'
  ltrue,           ///< Push 'true'
  lplus,           ///< increase logical depth
  lminus,          ///< decrease logical depth
  lnot,            ///< Unary negation
  lor,             ///< Binary logical OR
  land,            ///< Binary logical AND
  lend
};

struct LogicExpression {
  logic_int size_;
  logic_int *data_{nullptr};

  static VECCORE_ATT_HOST_DEVICE bool is_operator_token(logic_int lv) { return (lv >= lbegin); }

  VECCORE_ATT_HOST_DEVICE
  unsigned size() const { return (unsigned)size_; }

  VECCORE_ATT_HOST_DEVICE
  logic_int operator[](unsigned i) const { return data_[i]; }
};

///< Supported surface types
enum SurfaceType { kPlanar, kCylindrical, kConical, kSpherical, kTorus, kGenSecondOrder };

///< Supported frame types
///< kNoFrame     <- no frame, used for Inside only
///< kRangeZ      <- range along z-axis
///< kRing        <- a "ring" range on a plane
///< kZPhi        <- z and phi range on a cylinder
///< kRangeSph    <- theta and phi range on a sphere
///< kWindow      <- rectangular range in xy-plane
///< kTriangle    <- triangular range in xy-plane
enum FrameType { kNoFrame, kRangeZ, kRing, kZPhi, kRangeSph, kWindow, kTriangle, kQuadrilateral };

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
  Real_t radius{0}; ///< Cone radus at Z = 0: 0.5 * (rbottom + rup) Stored negative if flipped.
  Real_t slope{0};  ///< Cone slope  0.5 * (rup - rbottom)/dz --> for cyl extension this would be 0

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

///< Constants and tolerances
template <typename Real_t>
constexpr Real_t Tolerance()
{
  return 0;
}

template <>
constexpr double Tolerance()
{
  return 1.e-9;
}

template <>
constexpr float Tolerance()
{
  return 1.e-4;
}

template <typename Real_t>
bool ApproxEqual(Real_t t1, Real_t t2)
{
  return std::abs(t1 - t2) <= Tolerance<Real_t>();
}

template <typename Real_t>
bool ApproxEqualVector(Vector3D<Real_t> const &v1, Vector3D<Real_t> const &v2)
{
  return ApproxEqual(v1[0], v2[0]) && ApproxEqual(v1[1], v2[1]) && ApproxEqual(v1[2], v2[2]);
}

} // namespace vgbrep

#endif
