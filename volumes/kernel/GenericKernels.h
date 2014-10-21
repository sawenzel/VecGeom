/// \file GenericKernels.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_
#define VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "base/Vector3D.h"

namespace VECGEOM_NAMESPACE {

template <class Backend>
struct GenericKernels {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

  VECGEOM_CUDA_HEADER_BOTH
  static Float_t NormalizeAngle(Float_t angle) {
    return angle + kTwoPi*Float_t(((angle < 0.) - Int_t(angle * kTwoPiInv)));
  }

}; // End struct GenericKernels

template<bool tolerant, typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T MakePlusTolerant(T const &x) {
  return (tolerant) ? x+kTolerance : x;
}

template<bool tolerant, typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T MakeMinusTolerant(T const &x) {
  return (tolerant) ? x-kTolerance : x;
}

template <class Backend>
VECGEOM_INLINE
typename Backend::int_v FindSegmentIndex(
    Precision const array[],
    int size,
    typename Backend::precision_v const &element);

/// Brute force. Maybe binary search for large sizes?
template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int FindSegmentIndex<kScalar>(
    Precision const array[],
    const int size,
    Precision const &element) {
  int index = 0;
  while (index < size && element > array[index]) ++index;
  return index-1;
}

#ifdef VECGEOM_VC
/// Brute force with optional early returns
template <>
VECGEOM_INLINE
VcInt FindSegmentIndex<kVc>(
    Precision const array[],
    const int size,
    VcPrecision const &element) {
  VcInt index = -1;
  for (int i = 0; i < size; ++i) {
    MaskedAssign(element >= array[i], i, &index);
    // if (kVc::early_returns) {
    //   if (IsFull(index < i)) break;
    // }
  }
  return index;
}
#endif

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

template <bool flipSignT>
struct FlipSign {};

template <>
struct FlipSign<true> {
  template <class T>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static T Flip(T const &value) {
    return -value;
  }
};

template <>
struct FlipSign<false> {
  template <class T>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static T Flip(T const &value) {
    return value;
  }
};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_