#include "backend/cilk_backend.h"

namespace vecgeom {

const CilkBool      Impl<kCilk>::kTrue  = CilkBool(true);
const CilkBool      Impl<kCilk>::kFalse = CilkBool(false);
const CilkPrecision Impl<kCilk>::kOne   = CilkPrecision(1.0);
const CilkPrecision Impl<kCilk>::kZero  = CilkPrecision(0.0);

} // End namespace vecgeom