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
  typedef double precision_v;
  typedef bool   bool_v;
  constexpr static bool early_returns = true;
  constexpr static precision_v kOne = 1.0;
  constexpr static precision_v kZero = 0.0;
  constexpr static bool_v kTrue = true;
  constexpr static bool_v kFalse = false;
};

typedef Impl<kScalar>::int_v    ScalarInt;
typedef Impl<kScalar>::precision_v ScalarDouble;
typedef Impl<kScalar>::bool_v   ScalarBool;

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CondAssign(const bool cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  *output = (cond) ? thenval : elseval;
}

template <typename Type1, typename Type2>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void MaskedAssign(const bool cond,
                  Type1 const &thenval, Type2 *const output) {
  *output = (cond) ? thenval : *output;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Abs(const Type val) {
  return std::fabs(val);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Sqrt(const Type val) {
  return std::sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_