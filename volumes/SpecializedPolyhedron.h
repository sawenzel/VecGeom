/// \file SpecializedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "volumes/kernel/PolyhedronImplementation.h"
#include "volumes/PlacedPolyhedron.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
using SpecializedPolyhedron = ScalarShapeImplementationHelper< PolyhedronImplementation<innerRadiiT, phiCutoutT> >;


using SimplePolyhedron = SpecializedPolyhedron<Polyhedron::EInnerRadii::kGeneric,
   Polyhedron::EPhiCutout::kGeneric>;

} // End inline namespace

} // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
