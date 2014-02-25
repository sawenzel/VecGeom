#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "base/utilities.h"
#include "base/types.h"

namespace vecgeom {

template <>
struct Impl<kScalar> {
  typedef Precision precision;
  typedef int       int_v;
  typedef Precision precision_v;
  typedef bool      bool_v;
  const static bool early_returns = true;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
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
  return fabs(val);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Sqrt(const Type val) {
  return sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_