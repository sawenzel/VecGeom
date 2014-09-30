/// \file scalar/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "base/Global.h"

#include <algorithm>

namespace VECGEOM_NAMESPACE {

struct kScalar {
  typedef int       int_v;
  typedef Precision precision_v;
  typedef bool      bool_v;
  typedef Inside_t  inside_v;
  const static bool early_returns = true;
#ifdef VECGEOM_NVCC
  const static precision_v kOne = 1.0;
  const static precision_v kZero = 0.0;
#else
  constexpr static precision_v kOne = 1.0;
  constexpr static precision_v kZero = 0.0;
#endif
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static constexpr bool IsEqual() { return false; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision Convert(Precision const &input) { return input; }
};

template <>
VECGEOM_CUDA_HEADER_BOTH
inline constexpr bool kScalar::IsEqual<kScalar>() { return true; }

typedef kScalar::int_v    ScalarInt;
typedef kScalar::precision_v ScalarDouble;
typedef kScalar::bool_v   ScalarBool;

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CondAssign(const bool cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  *output = (cond) ? thenval : elseval;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void MaskedAssign(const bool cond,
                  Type const &thenval, Type *const output) {
  *output = (cond) ? thenval : *output;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool IsFull(bool const &cond){
    return cond;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool Any(bool const &cond) {
  return cond;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool IsEmpty(bool const &cond){
    return !cond;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Abs(const Type val) {
  return fabs(val);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Sqrt(const Type val) {
  return sqrt(val);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type ATan2(const Type y, const Type x) {
  if (x != 0) return  std::atan2(y, x);
  if (y >  0) return  kPi / 2;
  if (y <  0) return -kPi / 2;
  return  0;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T Min(T val1, T val2) {
#ifndef VECGEOM_NVCC
  return std::min(val1, val2);
#else
  return min(val1, val2);
#endif
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T Max(T val1, T val2) {
#ifndef VECGEOM_NVCC
  return std::max(val1, val2);
#else
  return max(val1, val2);
#endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision sin(const Precision radians) {
#ifndef VECGEOM_NVCC
  return std::sin(radians);
#else
  return sin(radians);
#endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision cos(const Precision radians) {
#ifndef VECGEOM_NVCC
  return std::cos(radians);
#else
  return cos(radians);
#endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision tan(const Precision radians) {
#ifndef VECGEOM_NVCC
  return std::tan(radians);
#else
  return atan(radians);
#endif
}

namespace {

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Swap(Type &a, Type &b) {
#ifndef VECGEOM_NVCC
  std::swap(a, b);
#else
  Type c = a; a = b; b = c;
#endif
}

}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void copy(Type const *begin, Type const *const end, Type *const target) {
#ifndef VECGEOM_NVCC
  std::copy(begin, end, target);
#else
  memcpy(target, begin, (end-begin)*sizeof(Type));
#endif
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void reverse_copy(Type const *const begin, Type const *end,
                  Type *const target) {
#ifndef VECGEOM_NVCC
  std::reverse_copy(begin, end, target);
#else
  while (end-- != begin) *(target++) = *end; 
#endif
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void reverse(Type *begin, Type *end) {
#ifndef VECGEOM_NVCC
  std::reverse(begin, end);
#else
  while (begin++ < end--) Swap(begin, end);
#endif
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type* AlignedAllocate(size_t size) {
#ifndef VECGEOM_NVCC
  return static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
#else
  return new Type[size];
#endif
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AlignedFree(Type *allocated) {
#ifndef VECGEOM_NVCC
  _mm_free(allocated);
#else
  delete[] allocated;
#endif
}

template <typename IteratorType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
IteratorType min_element(IteratorType first, IteratorType last) {
#ifndef VECGEOM_NVCC
  return std::min_element(first, last);
#else
  return min_element(first, last);
#endif
}

template <typename IteratorType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool all_of(IteratorType first, IteratorType last) {
#ifndef VECGEOM_NVCC
  return std::all_of(first, last, [](bool b){return b;});
#else
  while (first++ != last) if (!first) return false;
  return true;
#endif
}

template <typename InputIterator1, typename InputIterator2>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool equal(InputIterator1 first, InputIterator1 last, InputIterator2 target) {
#ifndef VECGEOM_NVCC
  return std::equal(first, last, target);
#else
  return equal(first, last, target);
#endif
}

} // End global namespace

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_
