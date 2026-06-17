/// @file SpecializedHalfSpace.h
/// @brief Specialized placed-volume alias for half-space navigation.

#ifndef VECGEOM_VOLUMES_SPECIALIZEDHALFSPACE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDHALFSPACE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedHalfSpace.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedHalfSpace.h"
#include "VecGeom/volumes/kernel/HalfSpaceImplementation.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedHalfSpace = SpecializedVolImplHelper<HalfSpaceImplementation>;
using SimpleHalfSpace      = SpecializedHalfSpace;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDHALFSPACE_H_
