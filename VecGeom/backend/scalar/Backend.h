/// \file scalar/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "VecGeom/base/Global.h"

#include <algorithm>
#include <cstring>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

struct kScalar {
  typedef int int_v;
  typedef Precision precision_v;
  typedef bool bool_v;
  typedef Inside_t inside_v;
  // alternative typedefs ( might supercede above typedefs )
  typedef int Int_t;
  typedef Precision Double_t;
  typedef bool Bool_t;
  typedef int Index_t; // the type of indices

  constexpr static precision_v kOne  = 1.0;
  constexpr static precision_v kZero = 0.0;
  const static bool_v kTrue          = true;
  const static bool_v kFalse         = false;

  template <class Backend>
  VECCORE_ATT_HOST_DEVICE static VECGEOM_CONSTEXPR_RETURN bool IsEqual()
  {
    return false;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision Convert(Precision const &input) { return input; }
};

template <>
VECCORE_ATT_HOST_DEVICE inline VECGEOM_CONSTEXPR_RETURN bool kScalar::IsEqual<kScalar>()
{
  return true;
}

typedef kScalar::int_v ScalarInt;
typedef kScalar::precision_v ScalarDouble;
typedef kScalar::bool_v ScalarBool;

#ifdef VECGEOM_SCALAR
constexpr size_t kVectorSize = 1;
#define VECGEOM_BACKEND_TYPE vecgeom::kScalar
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) (*(P))
#define VECGEOM_BACKEND_PRECISION_TYPE vecgeom::Precision
#define VECGEOM_BACKEND_PRECISION_TYPE_SIZE 1
//#define VECGEOM_BACKEND_PRECISION_NOT_SCALAR
#define VECGEOM_BACKEND_BOOL vecgeom::ScalarBool
#define VECGEOM_BACKEND_INSIDE vecgeom::kScalar::inside_v
#endif

// template <typename Type>
// VECGEOM_FORCE_INLINE
// VECCORE_ATT_HOST_DEVICE
// void swap(Type &a, Type &b)
//{
//  std::swap(a, b);
//}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void copy(Type const *begin, Type const *const end, Type *const target)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  std::copy(begin, end, target);
#else
  std::memcpy(target, begin, sizeof(Type) * (end - begin));
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Type *AlignedAllocate(size_t size)
{
#ifndef VECCORE_CUDA
  return static_cast<Type *>(vecCore::AlignedAlloc(kAlignmentBoundary, sizeof(Type) * size));
#else
  Type *ptr = new Type[size];
  assert(ptr != nullptr && "Error: Memory allocation failed! If on GPU, consider increasing the heap size on GPU with "
                           "CudaDeviceSetHeapLimit(new_size)");
  return ptr;
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AlignedFree(Type *allocated)
{
#ifndef VECCORE_CUDA
  vecCore::AlignedFree(allocated);
#else
  delete[] allocated;
#endif
}

template <typename InputIterator1, typename InputIterator2>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool equal(InputIterator1 first, InputIterator1 last,
                                                        InputIterator2 target)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  return std::equal(first, last, target);
#else
  while (first != last) {
    if (*first++ != *target++) return false;
  }
  return true;
#endif
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_
