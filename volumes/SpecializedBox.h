/// \file SpecializedBox.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "base/Global.h"

#include "volumes/kernel/BoxImplementation.h"
#include "volumes/PlacedBox.h"
#include "volumes/ShapeImplementationHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedBox = ShapeImplementationHelper<BoxImplementation<transCodeT, rotCodeT> >;

using SimpleBox = SpecializedBox<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
