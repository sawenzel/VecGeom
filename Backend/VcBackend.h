#ifndef VECGEOM_BACKEND_VCBACKEND_H_
#define VECGEOM_BACKEND_VCBACKEND_H_

#include <Vc/Vc>
#include "Base/Utilities.h"
#include "Base/Types.h"

template <>
struct Impl<kVc> {
  typedef Vc::int_v    int_v;
  typedef Vc::double_v double_v;
  typedef Vc::double_m bool_v;
  const static bool early_returns = false;
};

constexpr int kVectorSize = Impl<kVc>::double_v::Size;

typedef Impl<kVc>::int_v    VcInt;
typedef Impl<kVc>::double_v VcDouble;
typedef Impl<kVc>::bool_v   VcBool;

#endif // VECGEOM_BACKEND_VCBACKEND_H_