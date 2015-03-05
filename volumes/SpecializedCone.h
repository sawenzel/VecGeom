/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedCone.h"
#include "volumes/ScalarShapeImplementationHelper.h"
#include "base/SOA3D.h"
#include "volumes/PlacedBox.h"

#include <stdio.h>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename ConeType>
using SpecializedCone = ScalarShapeImplementationHelper<ConeImplementation<transCodeT, rotCodeT, ConeType>>;

using SimpleCone = SpecializedCone<translation::kGeneric, rotation::kGeneric, ConeTypes::UniversalCone>;
using SimpleUnplacedCone = SpecializedCone<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>;


} // End impl namespace
} // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

