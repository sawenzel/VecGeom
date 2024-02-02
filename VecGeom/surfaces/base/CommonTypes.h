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
VECCORE_ATT_HOST_DEVICE constexpr Real_t Tolerance()
{
  return 0;
}

template <>
VECCORE_ATT_HOST_DEVICE constexpr double Tolerance()
{
  return 1.e-9;
}

template <>
VECCORE_ATT_HOST_DEVICE constexpr float Tolerance()
{
  return 1.e-4;
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ApproxEqual(Real_t t1, Real_t t2)
{
  return std::abs(t1 - t2) <= Tolerance<Real_t>();
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ApproxEqualVector(Vector3D<Real_t> const &v1, Vector3D<Real_t> const &v2)
{
  return ApproxEqual(v1[0], v2[0]) && ApproxEqual(v1[1], v2[1]) && ApproxEqual(v1[2], v2[2]);
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ApproxEqualVector2(Vector2D<Real_t> const &v1, Vector2D<Real_t> const &v2)
{
  return ApproxEqual(v1[0], v2[0]) && ApproxEqual(v1[1], v2[1]);
}

/// @brief Truncate a value to as many significant digits as a given tolerance.
///  For example, if the tolerance is 1e-9, truncate the vlaue to 9 significant digits
/// @tparam Real_t Precision type
/// @param x Value to truncate
/// @param tolerance Tolerance with as many significant digits as the truncation result
/// @return Truncated value
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t TruncateValue(Real_t x, Real_t tolerance = Tolerance<Real_t>())
{
  auto div = std::abs(x);
  while (int(div) > 0) {
    div /= 10;
    tolerance *= 10;
  }
  return tolerance * std::lround(x / tolerance);
}

/// @brief Return the rounding error of a value, assuming as many significant digits as a provided tolerance
/// @tparam Real_t Precision type
/// @param x Value subject to rounding
/// @param tolerance Tolerance with as many significant digits as the truncation result
/// @return Truncation error
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t RoundingError(Real_t x, Real_t tolerance = Tolerance<Real_t>())
{
  auto div = std::abs(x);
  while (int(div) > 0) {
    div /= 10;
    tolerance *= 10;
  }
  return tolerance;
}
/// @brief Data for torus surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t, typename Real_s = Real_t>
struct TorusData {
  Real_t rTor{0};  ///< radius to the center the torus.
  Real_t rTube{0}; ///< radius of the tube around the torus center, if negative, the torus is flipped
  AngleVector<Real_t> vecSPhi{
      vecgeom::kInfLength,
      vecgeom::kInfLength}; ///< Cartesian coordinates of vectors that represents the start of the phi-cut.
  AngleVector<Real_t> vecEPhi{
      vecgeom::kInfLength,
      vecgeom::kInfLength}; ///< Cartesian coordinates of vectors that represents the end of the phi-cut.
  CylData<Real_t, Real_t> cycl_data;

  /// @brief Check if local point is in the phi range
  /// @param local Point in local coordinates
  /// @return Point inside phi
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsidePhi(Vector3D<Real_t> const &local) const
  {
    if (vecSPhi[0] >= vecgeom::kInfLength - vecgeom::kTolerance &&
        vecEPhi[0] >= vecgeom::kInfLength - vecgeom::kTolerance)
      return true;
    AngleVector<Real_t> localAngle{local[0], local[1]};
    auto convex = vecSPhi.CrossZ(vecEPhi) > Real_t(0);
    auto in1    = vecSPhi.CrossZ(localAngle) > -vecgeom::kTolerance;
    auto in2    = localAngle.CrossZ(vecEPhi) > -vecgeom::kTolerance;
    return convex ? in1 && in2 : in1 || in2;
  }

  TorusData() = default;
  TorusData(Real_s rad, Real_s rad_tube, Real_s sphi = Real_s{0}, Real_s ephi = Real_s{0}, bool flip = false)
      : rTor(rad), rTube(flip ? -rad_tube : rad_tube), vecSPhi(vecgeom::Cos(sphi), vecgeom::Sin(sphi)),
        vecEPhi(vecgeom::Cos(ephi), vecgeom::Sin(ephi)), cycl_data(rad + rad_tube, false){};
  VECCORE_ATT_HOST_DEVICE
  Real_s Radius() const { return std::abs(Real_s(rTor)); }
  VECCORE_ATT_HOST_DEVICE
  Real_s RadiusTube() const { return std::abs(Real_s(rTube)); }
  VECCORE_ATT_HOST_DEVICE
  VECCORE_ATT_HOST_DEVICE
  bool IsFlipped() const { return rTube < 0; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  CylData<Real_t, Real_t> const &GetCylData() const { return cycl_data; }
};

} // namespace vgbrep

#endif
