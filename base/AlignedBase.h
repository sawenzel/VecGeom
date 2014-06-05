/// \file AlignedBase.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ALIGNEDBASE_H_
#define VECGEOM_BASE_ALIGNEDBASE_H_

#include "base/Global.h"
#ifdef VECGEOM_VC
#include <Vc/Vc>
#endif

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_VC
class AlignedBase : public Vc::VectorAlignedBase {};
#else
class AlignedBase {};
#endif

} // End global namespace

#endif // VECGEOM_BASE_ALIGNEDBASE_H_