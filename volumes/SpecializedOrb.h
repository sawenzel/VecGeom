/// @file SpecializedOrb.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDORB_H_
#define VECGEOM_VOLUMES_SPECIALIZEDORB_H_

#include "base/Global.h"

#include "volumes/kernel/OrbImplementation.h"
#include "volumes/PlacedOrb.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedOrb = ShapeImplementationHelper<OrbImplementation<transCodeT, rotCodeT> >;

using SimpleOrb = SpecializedOrb<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDORB_H_
