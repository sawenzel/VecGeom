/**
 * @file scalar/backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include <algorithm>

#include "base/global.h"

namespace VECGEOM_NAMESPACE {

struct kScalar {
  typedef int       int_v;
  typedef Precision precision_v;
  typedef bool      bool_v;
  typedef Inside_t  inside_v;
  const static bool early_returns = true;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
};

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

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision Min(Precision const &val1, Precision const &val2) {
  #ifndef VECGEOM_NVCC
  return std::min(val1, val2);
  #else
  return min(val1, val2);
  #endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision Max(Precision const &val1, Precision const &val2) {
  #ifndef VECGEOM_NVCC
  return std::max(val1, val2);
  #else
  return max(val1, val2);
  #endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int Min(int const &val1, int const &val2) {
  #ifndef VECGEOM_NVCC
  return std::min(val1, val2);
  #else
  return min(val1, val2);
  #endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int Max(int const &val1, int const &val2) {
  #ifndef VECGEOM_NVCC
  return std::max(val1, val2);
  #else
  return max(val1, val2);
  #endif
}

} // End global namespace

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_