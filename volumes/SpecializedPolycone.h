/*
 * SpecializedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYCONE_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "volumes/kernel/PolyconeImplementation.h"
#include "volumes/PlacedPolycone.h"
#include "volumes/ScalarShapeImplementationHelper.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedPolycone = ScalarShapeImplementationHelper<PolyconeImplementation<transCodeT, rotCodeT> >;

using SimplePolycone = SpecializedPolycone<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif /* VECGEOM_VOLUMES_SPECIALIZEDPOLYCONE_H_ */
