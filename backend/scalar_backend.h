#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "base/utilities.h"
#include "base/types.h"
#include "backend/backend.h"

namespace vecgeom {

template <>
struct Impl<kScalar> {
  typedef double precision;
  typedef int    int_v;
  typedef double double_v;
  typedef bool   bool_v;
  constexpr static bool early_returns = true;
  constexpr static double_v kOne = 1.0;
  constexpr static double_v kZero = 0.0;
  constexpr static bool_v kTrue = true;
  constexpr static bool_v kFalse = false;
};

typedef Impl<kScalar>::int_v    ScalarInt;
typedef Impl<kScalar>::double_v ScalarDouble;
typedef Impl<kScalar>::bool_v   ScalarBool;

template <ImplType it = kScalar, typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CondAssign(typename Impl<it>::bool_v const &cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  *output = (cond) ? thenval : elseval;
}

template <ImplType it = kScalar, typename Type1, typename Type2>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void MaskedAssign(typename Impl<it>::bool_v const &cond,
                  Type1 const &thenval, Type2 *const output) {
  *output = (cond) ? thenval : *output;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
double Abs<kScalar, double>(double const &val) {
  return std::fabs(val);
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
float Abs<kScalar, float>(float const &val) {
  return std::fabs(val);
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
double Sqrt<kScalar, double>(double const &val) {
  return std::sqrt(val);
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
float Sqrt<kScalar, float>(float const &val) {
  return std::sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_