#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "Base/Utilities.h"
#include "Base/Types.h"

template <>
struct Impl<kScalar> {
  typedef int    int_v;
  typedef double double_v;
  typedef bool   bool_v;
  const static bool early_returns = true;
};

typedef Impl<kScalar>::int_v    ScalarInt;
typedef Impl<kScalar>::double_v ScalarDouble;
typedef Impl<kScalar>::bool_v   ScalarBool;

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_