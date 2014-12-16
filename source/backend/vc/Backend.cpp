/// \file vc/Backend.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch

#include "backend/vc/Backend.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const VcBool      kVc::kTrue  = VcBool(true);
const VcBool      kVc::kFalse = VcBool(false);
const VcPrecision kVc::kOne   = VcPrecision(1.); //Vc::One;
const VcPrecision kVc::kZero  = VcPrecision(0.); //Vc::Zero;

} } // End global namespace
