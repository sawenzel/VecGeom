/// \file PlacedTorus.cpp

#include "volumes/PlacedTorus2.h"
#include "volumes/Torus2.h"
#include "volumes/SpecializedTorus2.h"

#ifdef VECGEOM_ROOT
#include "TGeoTorus.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Torus.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTorus2::ConvertToUnspecialized() const {
  return new SimpleTorus2(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTorus2::ConvertToRoot() const {
  return new TGeoTorus(GetLabel().c_str(), rtor(), rmin(), rmax(),
          sphi()*kRadToDeg, dphi()*kRadToDeg);
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTorus2::ConvertToUSolids() const {
    return NULL;
    //  return new UTubs(GetLabel().c_str(), rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTorus2::ConvertToGeant4() const {
  return new G4Torus(GetLabel().c_str(), rmin(), rmax(), rtor(), sphi(), dphi());
}
#endif

#endif // VECGEOM_BENCHMARK

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedTorus2 )

#endif

} // End global namespace

