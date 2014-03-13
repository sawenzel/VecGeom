#include "backend/vc/backend.h"

namespace vecgeom {

const VcBool      Impl<kVc>::kTrue  = VcBool(true);
const VcBool      Impl<kVc>::kFalse = VcBool(false);
const VcPrecision Impl<kVc>::kOne   = VcPrecision(1.); //Vc::One;
const VcPrecision Impl<kVc>::kZero  = VcPrecision(0.); //Vc::Zero;

} // End namespace vecgeom
