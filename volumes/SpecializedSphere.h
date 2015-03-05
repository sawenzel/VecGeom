/// @file SpecializedSphere.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/kernel/SphereImplementation.h"
#include "volumes/PlacedSphere.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedSphere = ShapeImplementationHelper<SphereImplementation<transCodeT, rotCodeT> >;

using SimpleSphere = SpecializedSphere<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
