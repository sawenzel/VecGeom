/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "backend/cilk/backend.h"

namespace vecgeom {

const CilkBool      kCilk::kTrue  = CilkBool(true);
const CilkBool      kCilk::kFalse = CilkBool(false);
const CilkPrecision kCilk::kOne   = CilkPrecision(1.0);
const CilkPrecision kCilk::kZero  = CilkPrecision(0.0);

} // End namespace vecgeom