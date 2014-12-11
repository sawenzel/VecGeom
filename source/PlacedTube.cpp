/// \file PlacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/PlacedTube.h"
#include "volumes/Tube.h"
#include "volumes/SpecializedTube.h"

#ifdef VECGEOM_ROOT
#include "TGeoTube.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTubs.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Tubs.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTube::ConvertToUnspecialized() const {
  return new SimpleTube(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTube::ConvertToRoot() const {
  if(dphi() >= 2*M_PI)
     return new TGeoTube(GetLabel().c_str(), rmin(), rmax(), z());
  return new TGeoTubeSeg(GetLabel().c_str(), rmin(), rmax(), z(), sphi()*(180/M_PI), sphi()*(180/M_PI)+dphi()*(180/M_PI) );
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTube::ConvertToUSolids() const {
  return new UTubs(GetLabel().c_str(), rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTube::ConvertToGeant4() const {
  return new G4Tubs(GetLabel().c_str(), rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::UniversalTube )

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::NonHollowTube )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::NonHollowTubeWithSmallerThanPiSector )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::NonHollowTubeWithBiggerThanPiSector )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::NonHollowTubeWithPiSector )

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::HollowTube )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::HollowTubeWithSmallerThanPiSector )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::HollowTubeWithBiggerThanPiSector )
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( SpecializedTube, TubeTypes::HollowTubeWithPiSector )
#endif

#endif

} // End global namespace  

