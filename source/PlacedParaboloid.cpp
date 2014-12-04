/// \file PlacedParaboloid.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/PlacedParaboloid.h"

#include "volumes/Paraboloid.h"

#ifndef VECGEOM_NVCC

#ifdef VECGEOM_ROOT
#include "TGeoParaboloid.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Paraboloid.hh"
#endif

#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif

#endif // VECGEOM_NVCC

#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedParaboloid::ConvertToUnspecialized() const {
    return new SimpleParaboloid(GetLabel().c_str(), logical_volume(),transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedParaboloid::ConvertToRoot() const {
    return new TGeoParaboloid(GetLabel().c_str(), GetRlo(), GetRhi(), GetDz());
    
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedParaboloid::ConvertToUSolids() const {
    std::cerr << "**************************************************************\n";
    std::cerr << "WARNING: Paraboloid unsupported for USolids.; returning a box\n";
    std::cerr << "**************************************************************\n";
    return new UBox("",10,10,10);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedParaboloid::ConvertToGeant4() const {
    return new G4Paraboloid(GetLabel(), GetDz(), GetRlo(), GetRhi());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedParaboloid )

#endif // VECGEOM_NVCC

} // End global namespace
