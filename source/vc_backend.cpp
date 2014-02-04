#include "backend/vc_backend.h"

namespace vecgeom {

const VcBool   Impl<kVc>::kTrue  = VcBool(true);
const VcBool   Impl<kVc>::kFalse = VcBool(false);
const VcDouble Impl<kVc>::kOne   = Vc::One;
const VcDouble Impl<kVc>::kZero  = Vc::Zero;

} // End namespace vecgeom