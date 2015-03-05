/// \file SpecializedParallelepiped.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "volumes/kernel/ParallelepipedImplementation.h"
#include "volumes/PlacedParallelepiped.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedParallelepiped = ShapeImplementationHelper<ParallelepipedImplementation<transCodeT, rotCodeT> >;

using SimpleParallelepiped = SpecializedParallelepiped<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
