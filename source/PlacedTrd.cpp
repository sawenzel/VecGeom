/// @file PlacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/Trd.h"

#ifndef VECGEOM_NVCC

#ifdef VECGEOM_ROOT
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTrd.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Trd.hh"
#endif

#endif // VECGEOM_NVCC

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTrd::ConvertToUnspecialized() const {
  return new SimpleTrd(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTrd::ConvertToRoot() const {
  if(dy1() == dy2())
    return new TGeoTrd1(GetLabel().c_str(), dx1(), dx2(), dy1(), dz());
  return new TGeoTrd2(GetLabel().c_str(), dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTrd::ConvertToUSolids() const {
  return new UTrd("", dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTrd::ConvertToGeant4() const {
  return new G4Trd(GetLabel(), dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#endif // VECGEOM_NVCC

} // End im%pl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTrd, TrdTypes::UniversalTrd )

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTrd, TrdTypes::Trd1 )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTrd, TrdTypes::Trd2 )

#endif

} // End global namespace

