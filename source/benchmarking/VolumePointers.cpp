/// \file VolumePointers.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "benchmarking/VolumePointers.h"
#include "volumes/PlacedVolume.h"
#include <iostream>

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif
#ifdef VECGEOM_USOLIDS
#include "VUSolid.hh"
#endif
#ifdef VECGEOM_GEANT4
#include "G4VSolid.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VolumePointers::VolumePointers(VPlacedVolume const *const volume)
    : fSpecialized(volume), fUnspecialized(NULL),
#ifdef VECGEOM_ROOT
      fRoot(NULL),
#endif
#ifdef VECGEOM_USOLIDS
      fUSolids(NULL),
#endif
#ifdef VECGEOM_GEANT4
      fGeant4(NULL),
#endif
      fInitial(kBenchmarkSpecialized) {
  ConvertVolume();
}

VolumePointers::VolumePointers(VolumePointers const &other)
    : fSpecialized(other.fSpecialized), fUnspecialized(NULL),
#ifdef VECGEOM_ROOT
      fRoot(NULL),
#endif
#ifdef VECGEOM_USOLIDS
      fUSolids(NULL),
#endif
#ifdef VECGEOM_GEANT4
      fGeant4(NULL),
#endif
      fInitial(other.fInitial) {
  ConvertVolume();
}

VolumePointers::~VolumePointers() {
  Deallocate();
}

VolumePointers& VolumePointers::operator=(VolumePointers const &other) {
  this->Deallocate();
  this->fSpecialized = other.fSpecialized;
  this->ConvertVolume();
  return *this;
}

void VolumePointers::ConvertVolume() {
  if (!fUnspecialized) fUnspecialized = fSpecialized->ConvertToUnspecialized();
#ifdef VECGEOM_ROOT
  if (!fRoot)          fRoot          = fSpecialized->ConvertToRoot();
#endif
#ifdef VECGEOM_USOLIDS
  if (!fUSolids)       fUSolids       = fSpecialized->ConvertToUSolids();
#endif
#ifdef VECGEOM_GEANT4
  if (!fGeant4)        fGeant4        = fSpecialized->ConvertToGeant4();
#endif
}

void VolumePointers::Deallocate() {
  /*
  if (fInitial != kBenchmarkSpecialized)   delete fSpecialized;
  if (fInitial != kBenchmarkUnspecialized) delete fUnspecialized;
>>>>>>> master
#ifdef VECGEOM_ROOT
 // if (initial_ != kBenchmarkRoot)          delete root_;
#endif
#ifdef VECGEOM_USOLIDS
 // if (initial_ != kBenchmarkUSolids)       delete usolids_;
#endif
#ifdef VECGEOM_GEANT4
  //  if (fInitial != kBenchmarkGeant4)        delete fGeant4;
#endif
  */
}

} // End impl namespace
} // End global namespace
