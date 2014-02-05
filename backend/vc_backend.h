#ifndef VECGEOM_BACKEND_VCBACKEND_H_
#define VECGEOM_BACKEND_VCBACKEND_H_

#include <Vc/Vc>
#include "base/utilities.h"
#include "base/types.h"
#include "backend/backend.h"

namespace vecgeom {

template <>
struct Impl<kVc> {
  typedef double       precision;
  typedef Vc::int_v    int_v;
  typedef Vc::double_v double_v;
  typedef Vc::double_m bool_v;
  constexpr static bool early_returns = false;
  const static double_v kOne;
  const static double_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
};

constexpr int kVectorSize = Impl<kVc>::double_v::Size;

typedef Impl<kVc>::int_v    VcInt;
typedef Impl<kVc>::double_v VcDouble;
typedef Impl<kVc>::bool_v   VcBool;

template <ImplType it = kVc, typename Type>
VECGEOM_INLINE
void CondAssign(typename Impl<it>::bool_v const &cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  (*output)(cond) = thenval;
  (*output)(!cond) = elseval;
}

template <ImplType it = kVc, typename Type1, typename Type2>
VECGEOM_INLINE
void MaskedAssign(typename Impl<it>::bool_v const &cond,
                  Type1 const &thenval, Type2 *const output) {
  (*output)(cond) = thenval;
}

template <>
VECGEOM_INLINE
VcDouble Abs<kVc, VcDouble>(VcDouble const &val) {
  return Vc::abs(val);
}

template <>
VECGEOM_INLINE
VcDouble Sqrt<kVc, VcDouble>(VcDouble const &val) {
  return Vc::sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_VCBACKEND_H_