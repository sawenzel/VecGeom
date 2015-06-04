#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H
#define VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H

#include "base/Global.h"

#include "volumes/kernel/BooleanImplementation.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/PlacedBooleanVolume.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <BooleanOperation boolOp, TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedBooleanVolume = ScalarShapeImplementationHelper<BooleanImplementation<boolOp, transCodeT, rotCodeT> >;

using GenericPlacedUnionVolume = SpecializedBooleanVolume<kUnion, translation::kGeneric, rotation::kGeneric>;
using GenericPlacedIntersectionVolume = SpecializedBooleanVolume<kIntersection, translation::kGeneric, rotation::kGeneric>;
using GenericPlacedSubtractionVolume = SpecializedBooleanVolume<kSubtraction, translation::kGeneric, rotation::kGeneric>;

using GenericUnionVolume = SpecializedBooleanVolume<kUnion, translation::kIdentity, rotation::kIdentity>;
using GenericIntersectionVolume = SpecializedBooleanVolume<kIntersection, translation::kIdentity, rotation::kIdentity>;
using GenericSubtractionVolume = SpecializedBooleanVolume<kSubtraction, translation::kIdentity, rotation::kIdentity>;

} // End impl namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H
