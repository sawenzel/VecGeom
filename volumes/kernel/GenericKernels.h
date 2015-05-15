/// \file GenericKernels.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_
#define VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "base/Vector3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Backend>
struct GenericKernels {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

}; // End struct GenericKernels

template<bool tolerant, typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T MakePlusTolerant(T const &x)
{
  return (tolerant)? x+kHalfTolerance : x;
}

template<bool tolerant, typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T MakeMinusTolerant(T const &x)
{
  return (tolerant)? x-kHalfTolerance : x;
}

template<bool tolerant, typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T  MakePlusTolerantSquare(T const &x, T const& xsq)
{
  return (tolerant)? xsq+kTolerance*x : xsq;
}

template<bool tolerant, typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T MakeMinusTolerantSquare(T const &x, T const &xsq)
{
  return (tolerant)? xsq-kTolerance*x : xsq;
}


template <bool treatSurfaceT, class Backend> struct TreatSurfaceTraits;
template <class Backend> struct TreatSurfaceTraits<true, Backend> {
  typedef typename Backend::inside_v Surface_t;
  static const Inside_t kInside = 0;
  static const Inside_t kOutside = 2;
};
template <class Backend> struct TreatSurfaceTraits<false, Backend> {
  typedef typename Backend::bool_v Surface_t;
  static const bool kInside = true;
  static const bool kOutside = false;
};

/// \brief Flips the sign of an input value depending on the set template
///        parameter.
template <bool flipT>
struct Flip;

template <>
struct Flip<true> {
  template <class T>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static T FlipSign(T const &value) {
    return -value;
  }
  template <class T>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static T FlipLogical(T const &value) {
    return !value;
  }
};

template <>
struct Flip<false> {
  template <class T>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static T FlipSign(T const &value) {
    return value;
  }
  template <class T>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static T FlipLogical(T const &value) {
    return value;
  }
};

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v NormalizeAngle(typename Backend::precision_v a) {
  return a + kTwoPi*((a<0)-typename Backend::int_v(a/kTwoPi));
}

// \param corner0 First corner of line segment.
// \param corner1 Second corner of line segment.
// \return Shortest distance from the point to the three dimensional line
//         segment represented by the two input points.
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v DistanceToLineSegmentSquared(
    Vector3D<Precision> corner0,
    Vector3D<Precision> corner1,
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t result(kInfinity);

  // Shortest distance is to corner of segment
  Vector3D<Precision> line = corner1 - corner0;
  Vector3D<Float_t> projection = point - corner0;
  Float_t dot0 = projection.Dot(line);
  Bool_t condition = dot0 <= 0;
  MaskedAssign(condition, (point - corner0).Mag2(), &result);
  if (IsFull(condition)) return result;
  Precision dot1 = line.Mag2();
  condition = dot1 <= dot0;
  MaskedAssign(condition, (point - corner1).Mag2(), &result);
  condition = result < kInfinity;
  if (IsFull(condition)) return result;

  // Shortest distance is to point on segment
  MaskedAssign(!condition, ((corner0 + (dot0 / dot1)*line) - point).Mag2(),
               &result);

  return result;
}

} // End inline namespace

} // End global namespace



#endif // VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_
