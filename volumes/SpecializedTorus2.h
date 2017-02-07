/// \file SpecializedTorus.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTORUS2_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTORUS2_H_

#include "base/Global.h"

#include "volumes/kernel/TorusImplementation2.h"
#include "volumes/PlacedTorus2.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

  // NOTE: we may want to specialize the torus like we do for the tube
  // at the moment this is not done

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTorus2 = ScalarShapeImplementationHelper<TorusImplementation2<transCodeT, rotCodeT> >;

using SimpleTorus2 = SpecializedTorus2<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTORUS_H_
