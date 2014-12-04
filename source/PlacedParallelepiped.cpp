/// @file PlacedParallelepiped.cpp
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedParallelepiped.h"

#include "volumes/Parallelepiped.h"

#ifdef VECGEOM_ROOT
#include "TGeoPara.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Para.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedParallelepiped::ConvertToUnspecialized() const {
  return new SimpleParallelepiped(GetLabel().c_str(), logical_volume(),
                                  transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedParallelepiped::ConvertToRoot() const {
  return new TGeoPara(GetLabel().c_str(), GetX(), GetY(), GetZ(), GetAlpha(),
                      GetTheta(), GetPhi());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedParallelepiped::ConvertToUSolids() const {
  assert(0 && "Parallelepiped unsupported for USolids.");
  return NULL;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedParallelepiped::ConvertToGeant4() const {
  return new G4Para(GetLabel(), GetX(), GetY(), GetZ(), GetAlpha(), GetTheta(),
                    GetPhi());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedParallelepiped )

#endif // VECGEOM_NVCC

} // End global namespace
