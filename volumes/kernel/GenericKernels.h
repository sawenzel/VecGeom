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
T
VECGEOM_CUDA_HEADER_BOTH
MakePlusTolerant( T const & x  )
{
    return (tolerant)? x+kTolerance : x;
}

template<bool tolerant, typename T>
T
VECGEOM_CUDA_HEADER_BOTH
MakeMinusTolerant( T x  )
{
    return (tolerant)? x-kTolerance : x;
}

template <class Backend>
VECGEOM_INLINE
typename Backend::int_v FindSegmentIndex(
    Precision const array[],
    int size,
    typename Backend::precision_v const &element);

/// Perform binary search
template <>
VECGEOM_INLINE
int FindSegmentIndex<kScalar>(
    Precision const array[],
    const int size,
    Precision const &element) {
  int middle;
  int start = 0;
  int end = size-1;
  do {
    middle = start + (end-start)/2;
    if (element < array[middle]) {
      end = middle-1;
      continue;
    }
    if (element > array[middle]) {
      start = middle+1;
      continue;
    }
    break;
  } while (start < end);
  return middle;
}

#ifdef VECGEOM_CUDA
/// Brute force to avoid uncoalesced memory access
template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int FindSegmentIndex<kCuda>(
    Precision const array[],
    const int size,
    Precision const &element) {
  int index = 0;
  while (index < size && element < array[index]) ++index;
  return index-1;
}
#endif

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
    if (kVc::early_returns) {
      if (IsFull(index < i)) break;
    }
  }
  return index;
}
#endif

/// Trait class to check if the projection of a point along the normal of the
/// plane has the correct sign, depending on whether the point is inside or
/// outside. For points behind the plane the distance should be positive (along
/// the normal), and opposite for points in front of the plane.
template <bool pointInsideT, class Backend>
class PointInsideTraits {
public:
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v IsValid(
      typename Backend::precision_v const &candidate);
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v Sign(
      typename Backend::precision_v const &candidate);
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v IsBetterDistance(
      typename Backend::precision_v const &candidate,
      typename Backend::precision_v const &current);
};
template <class Backend>
class PointInsideTraits<true, Backend> {
public:
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v IsValid(
      typename Backend::precision_v const &candidate) {
    return candidate >= 0;
  }
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v Sign(
      typename Backend::precision_v const &candidate) {
    return -candidate;
  }
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v IsBetterDistance(
      typename Backend::precision_v const &candidate,
      typename Backend::precision_v const &current) {
    return IsValid(candidate) && candidate < current;
  }
};
template <class Backend>
class PointInsideTraits<false, Backend> {
public:
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v IsValid(
      typename Backend::precision_v const &candidate) {
    return candidate <= 0;
  }
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v Sign(
      typename Backend::precision_v const &candidate) {
    return candidate;
  }
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v IsBetterDistance(
      typename Backend::precision_v const &candidate,
      typename Backend::precision_v const &current) {
    return IsValid(candidate) && candidate > current;
  }
};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_GENERICKERNELS_H_
