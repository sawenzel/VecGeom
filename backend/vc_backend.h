#ifndef VECGEOM_BACKEND_VCBACKEND_H_
#define VECGEOM_BACKEND_VCBACKEND_H_

#include <Vc/Vc>
#include "base/utilities.h"
#include "base/types.h"
#include "backend/backend.h"

namespace vecgeom {

template <>
struct Impl<kVc> {
  typedef Vc::int_v                   int_v;
  typedef Vc::Vector<Precision>       precision_v;
  typedef Vc::Vector<Precision>::Mask bool_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
};

constexpr int kVectorSize = Impl<kVc>::precision_v::Size;

typedef Impl<kVc>::int_v       VcInt;
typedef Impl<kVc>::precision_v VcPrecision;
typedef Impl<kVc>::bool_v      VcBool;

template <typename Type>
VECGEOM_INLINE
void CondAssign(VcBool const &cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  (*output)(cond) = thenval;
  (*output)(!cond) = elseval;
}

template <typename Type1, typename Type2>
VECGEOM_INLINE
void MaskedAssign(VcBool const &cond,
                  Type1 const &thenval, Type2 *const output) {
  (*output)(cond) = thenval;
}

VECGEOM_INLINE
VcPrecision Abs(VcPrecision const &val) {
  return Vc::abs(val);
}

VECGEOM_INLINE
VcPrecision Sqrt(VcPrecision const &val) {
  return Vc::sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_VCBACKEND_H_