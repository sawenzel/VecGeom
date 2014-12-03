/// @file SpecializedParaboloid.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_

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

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
