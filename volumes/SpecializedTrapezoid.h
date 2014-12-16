/// \file   SpecializedTrapezoid.h
/// \author Guilherme Lima (lima 'at' fnal 'dot' gov)
/*
 * 2014-05-01 - Created, based on the Parallelepiped draft
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_

#include "base/Global.h"

#include "volumes/kernel/TrapezoidImplementation.h"
#include "volumes/PlacedTrapezoid.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTrapezoid = ShapeImplementationHelper<TrapezoidImplementation<transCodeT, rotCodeT> >;

using SimpleTrapezoid = SpecializedTrapezoid<translation::kGeneric, rotation::kGeneric>;


} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_
