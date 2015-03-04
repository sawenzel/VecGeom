/// @file PlacedOrb.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedOrb.h"
#include "volumes/Orb.h"
#include "base/AOS3D.h"
#include "base/SOA3D.h"

#ifdef VECGEOM_USOLIDS
#include "UOrb.hh"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Orb.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedOrb::ConvertToUnspecialized() const {
  return new SimpleOrb(GetLabel().c_str(), GetLogicalVolume(),
                                  GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedOrb::ConvertToRoot() const {
  return new TGeoSphere(GetLabel().c_str(),0,GetRadius());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedOrb::ConvertToUSolids() const {

return new UOrb(GetLabel().c_str(),GetRadius());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedOrb::ConvertToGeant4() const {
return new G4Orb(GetLabel().c_str(),GetRadius());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedOrb )

#endif // VECGEOM_NVCC

} // End global namespace
