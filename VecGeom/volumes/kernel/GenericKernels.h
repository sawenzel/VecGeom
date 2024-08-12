/// \file GenericKernels.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_
#define VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Backend>
struct GenericKernels {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

}; // End struct GenericKernels

// typesafe sign
template <typename Real_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE constexpr int kSign(Real_t x)
{
  return (Real_t(0) < x) - (x < Real_t(0));
}

// relative tolerance specializations
template <typename Real_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE constexpr Real_t kRelTolerance(Real_t x,
                                                                            Real_t tolerance = kToleranceStrict<Real_t>)
{
  return Real_t(0);
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE constexpr double kRelTolerance<double>(double x, double tolerance)
{
  // If x is fractional, we don't want to reduce the tolerance
  return (x + kSign(x)) * tolerance;
}
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE constexpr float kRelTolerance<float>(float x, float tolerance)
{
  // If x is fractional, we don't want to reduce the tolerance
  return (x + kSign(x)) * tolerance;
}

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE constexpr T MakePlusTolerantRel(T const x,
                                                                             T tolerance = kToleranceStrict<T>)
{
  return (x + kRelTolerance<T>(x, tolerance));
}

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE constexpr T MakeMinusTolerantRel(T const x,
                                                                              T tolerance = kToleranceStrict<T>)
{
  return (x - kRelTolerance<T>(x, tolerance));
}

template <bool tolerant, typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T MakePlusTolerant(T const &x, vecCore::Scalar<T> tol = kToleranceDist<T>)
{
  return (tolerant) ? x + tol : x;
}

template <bool tolerant, typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T MakeMinusTolerant(T const &x, vecCore::Scalar<T> tol = kToleranceDist<T>)
{
  return (tolerant) ? x - T(tol) : x;
}

/// @brief Utilities to compute tolerance value for cross products. Length should be an overestimate
/// of the point vector length
/// P = point to check if on one side or the other of the AB segment. Cross product computed as:
///   cross = AP x AB
/// The distance from point P to segment AB is cross/|AB|, hence the tolerance of cross is kTolerance * |AB|
/// One needs to pass length > |AB| to the method.
template <bool tolerant, typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T MakePlusTolerantCrossProduct(T const &x, T const &length,
                                                                            vecCore::Scalar<T> tol = kToleranceDist<T>)
{
  return (tolerant) ? x + length * T(tol) : x;
}

template <bool tolerant, typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T MakeMinusTolerantCrossProduct(T const &x, T const &length,
                                                                             vecCore::Scalar<T> tol = kToleranceDist<T>)
{
  return (tolerant) ? x - length * T(tol) : x;
}

/// @brief Utility to compute (x + tol)^2 for proper account of tolerances when comparing squares.
template <bool tolerant, typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T MakePlusTolerantSquare(T const &x,
                                                                      vecCore::Scalar<T> tol = kToleranceDist<T>)
{
  // calculate (x + halftol) * (x + halftol) which should always >= 0;
  // in order to be fast, we neglect the + tol * tol term (since it should be negligible)
  return (tolerant) ? x * (x + T(2.0 * tol)) : x * x;
}

/// @brief Utility to compute (x - tol)^2 for proper account of tolerances when comparing squares.
template <bool tolerant, typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T MakeMinusTolerantSquare(T const &x,
                                                                       vecCore::Scalar<T> tol = kToleranceDist<T>)
{
  // calculate (x - halftol) * (x - halftol) which should always >= 0;
  // in order to be fast, we neglect the + tol * tol term (since it should be negligible)
  // but we make sure that there is never a negative sign (hence the Abs)
  return (tolerant) ? Abs(x * (x - T(2.0 * tol))) : x * x;
}

template <bool treatSurfaceT, class Backend>
struct TreatSurfaceTraits;
template <class Backend>
struct TreatSurfaceTraits<true, Backend> {
  typedef typename Backend::inside_v Surface_t;
  static const Inside_t kInside  = 0;
  static const Inside_t kOutside = 2;
};
template <class Backend>
struct TreatSurfaceTraits<false, Backend> {
  typedef typename Backend::bool_v Surface_t;
  static const bool kInside  = true;
  static const bool kOutside = false;
};

/// \brief Flips the sign of an input value depending on the set template
///        parameter.
template <bool flipT>
struct Flip;

template <>
struct Flip<true> {
  template <class T>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static T FlipSign(T const &value)
  {
    return -value;
  }
  template <class T>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static T FlipLogical(T const &value)
  {
    return !value;
  }
};

template <>
struct Flip<false> {
  template <class T>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static T FlipSign(T const &value)
  {
    return value;
  }
  template <class T>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static T FlipLogical(T const &value)
  {
    return value;
  }
};

template <class Backend>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename Backend::precision_v NormalizeAngle(
    typename Backend::precision_v a)
{
  return a + kTwoPi * ((a < 0) - typename Backend::int_v(a / kTwoPi));
}

// \param corner0 First corner of line segment.
// \param corner1 Second corner of line segment.
// \return Shortest distance from the point to the three dimensional line
//         segment represented by the two input points.
template <class Backend>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename Backend::precision_v DistanceToLineSegmentSquared(
    Vector3D<Precision> corner0, Vector3D<Precision> corner1, Vector3D<typename Backend::precision_v> const &point)
{

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t result(kInfLength);

  // Shortest distance is to corner of segment
  Vector3D<Precision> line     = corner1 - corner0;
  Vector3D<Float_t> projection = point - corner0;
  Float_t dot0                 = projection.Dot(line);
  Bool_t condition             = dot0 <= 0;
  vecCore__MaskedAssignFunc(result, condition, (point - corner0).Mag2());
  if (vecCore::MaskFull(condition)) return result;
  Precision dot1 = line.Mag2();
  condition      = dot1 <= dot0;
  vecCore__MaskedAssignFunc(result, condition, (point - corner1).Mag2());
  condition = result < kInfLength;
  if (vecCore::MaskFull(condition)) return result;

  // Shortest distance is to point on segment
  vecCore__MaskedAssignFunc(result, !condition, ((corner0 + (dot0 / dot1) * line) - point).Mag2());

  return result;
}

// \param corner0 First corner of line segment.
// \param corner1 Second corner of line segment.
// \return Shortest distance from the point to the three dimensional line
//         segment represented by the two input points.
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToLineSegmentSquared1(Vector3D<Precision> corner0,
                                                                                  Vector3D<Precision> corner1,
                                                                                  Vector3D<Real_v> const &point)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v result = InfinityLength<Real_v>();

  // Shortest distance is to corner of segment
  Vector3D<Precision> line    = corner1 - corner0;
  Vector3D<Real_v> projection = point - corner0;
  Real_v dot0                 = projection.Dot(line);
  Bool_v condition            = dot0 <= 0;
  vecCore__MaskedAssignFunc(result, condition, (point - corner0).Mag2());
  if (vecCore::MaskFull(condition)) return result;
  Precision dot1 = line.Mag2();
  condition      = dot1 <= dot0;
  vecCore__MaskedAssignFunc(result, condition, (point - corner1).Mag2());
  condition = result < kInfLength;
  if (vecCore::MaskFull(condition)) return result;

  // Shortest distance is to point on segment
  vecCore__MaskedAssignFunc(result, !condition, Real_v(((corner0 + (dot0 / dot1) * line) - point).Mag2()));

  return result;
}

// \param corner0 First corners of line segments.
// \param corner1 Second corners of line segments.
// \return Shortest distance from the point to the three dimensional set of line
//         segments represented by the input corners.
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToLineSegmentSquared2(Vector3D<Real_v> const &corner0,
                                                                                  Vector3D<Real_v> const &corner1,
                                                                                  Vector3D<Real_v> const &point,
                                                                                  vecCore::Mask_v<Real_v> const &mask)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v result = InfinityLength<Real_v>();

  // Shortest distance is to corner of segment
  Vector3D<Real_v> line       = corner1 - corner0;
  Vector3D<Real_v> projection = point - corner0;
  Real_v dot0                 = projection.Dot(line);
  Bool_v condition            = dot0 <= 0 && mask;
  vecCore__MaskedAssignFunc(result, condition, (point - corner0).Mag2());
  if (vecCore::MaskFull(condition && mask)) return result;
  Real_v dot1 = line.Mag2();
  condition   = dot1 <= dot0 && mask;
  vecCore__MaskedAssignFunc(result, condition, (point - corner1).Mag2());
  condition = result < kInfLength;
  if (vecCore::MaskFull(condition && mask)) return result;

  // Shortest distance is to point on segment
  vecCore__MaskedAssignFunc(result, !condition && mask, Real_v(((corner0 + (dot0 / dot1) * line) - point).Mag2()));

  return result;
}

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_
