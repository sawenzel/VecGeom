/// @file SpecializedParallelepiped.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

#include "base/global.h"

#include "volumes/kernel/ParallelepipedKernel.h"
#include "volumes/PlacedParallelepiped.h"
#include "volumes/ShapeImplementationHelper.h"

namespace VECGEOM_NAMESPACE {

template <class ParallelepipedSpecialization>
class SpecializedParallelepiped
    : public ShapeImplementationHelper<PlacedParallelepiped,
                                       ParallelepipedSpecialization> {};

typedef SpecializedParallelepiped<ParallelepipedSpecialization<
          true, true, true, translation::kIdentity, rotation::kIdentity>
        >SimpleParallelepiped;

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_