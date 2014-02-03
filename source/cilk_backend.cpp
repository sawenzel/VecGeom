#include "backend/cilk_backend.h"

namespace vecgeom {

const CilkBool   Impl<kCilk>::kTrue  = CilkBool(true);
const CilkBool   Impl<kCilk>::kFalse = CilkBool(false);
const CilkDouble Impl<kCilk>::kOne   = CilkDouble(1.0);
const CilkDouble Impl<kCilk>::kZero  = CilkDouble(0.0);

} // End namespace vecgeom