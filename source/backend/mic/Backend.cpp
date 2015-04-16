#pragma offload_attribute(push, target(mic))
#include "backend/mic/Backend.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const MicBool kMic::kTrue  = MicBool(0xffff);
const MicBool kMic::kFalse = MicBool(0x0);
const MicPrecision kMic::kOne  = MicPrecision(1.0);
const MicPrecision kMic::kZero = MicPrecision(0.0);

}
}
#pragma offload_attribute(pop)
