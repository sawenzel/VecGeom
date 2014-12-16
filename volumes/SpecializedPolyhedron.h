/// \file SpecializedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_

#include "base/Global.h"

#include "volumes/kernel/PolyhedronImplementation.h"
#include "volumes/PlacedPolyhedron.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <UnplacedPolyhedron::EInnerRadii innerRadiiT, UnplacedPolyhedron::EPhiCutout phiCutoutT>
using SpecializedPolyhedron = ScalarShapeImplementationHelper< PolyhedronImplementation<innerRadiiT, phiCutoutT> >;


using SimplePolyhedron = SpecializedPolyhedron<UnplacedPolyhedron::kInnerRadiiGeneric,
                                UnplacedPolyhedron::kPhiCutoutGeneric>;

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
