/// @file SpecializedParaboloid.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "volumes/kernel/ParaboloidImplementation.h"
#include "volumes/PlacedParaboloid.h"
#include "volumes/ShapeImplementationHelper.h"
#include "base/Transformation3D.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedParaboloid = ShapeImplementationHelper<ParaboloidImplementation<transCodeT, rotCodeT> >;

using SimpleParaboloid = SpecializedParaboloid<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
