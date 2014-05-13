/**
 * @file volume_pointers.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "management/volume_pointers.h"
#include "volumes/placed_volume.h"

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif // VECGEOM_ROOT
#ifdef VECGEOM_USOLIDS
#include "VUSolid.hh"
#endif // VECGEOM_USOLIDS

namespace VECGEOM_NAMESPACE {

VolumePointers::VolumePointers(VPlacedVolume const *const volume)
    : specialized_(volume), unspecialized_(NULL),
#ifdef VECGEOM_ROOT
      root_(NULL),
#endif
#ifdef VECGEOM_USOLIDS
      usolids_(NULL),
#endif
      initial_(kBenchmarkSpecialized) {
  ConvertVolume();
}

VolumePointers::VolumePointers(VolumePointers const &other)
    : specialized_(other.specialized_), unspecialized_(NULL),
#ifdef VECGEOM_ROOT
      root_(NULL),
#endif
#ifdef VECGEOM_USOLIDS
      usolids_(NULL),
#endif
      initial_(other.initial_) {
  ConvertVolume();
}

VolumePointers::~VolumePointers() {
  Deallocate();
}

VolumePointers& VolumePointers::operator=(VolumePointers const &other) {
  this->Deallocate();
  this->specialized_ = other.specialized_;
  this->ConvertVolume();
  return *this;
}

void VolumePointers::ConvertVolume() {
  if (!unspecialized_) unspecialized_ = specialized_->ConvertToUnspecialized();
#ifdef VECGEOM_ROOT
  if (!root_)          root_          = specialized_->ConvertToRoot();
#endif
#ifdef VECGEOM_USOLIDS
  if (!usolids_)       usolids_       = specialized_->ConvertToUSolids();
#endif
}

void VolumePointers::Deallocate() {
  if (initial_ != kBenchmarkSpecialized)   delete specialized_;
  if (initial_ != kBenchmarkUnspecialized) delete unspecialized_;
#ifdef VECGEOM_ROOT
  if (initial_ != kBenchmarkRoot)          delete root_;
#endif
#ifdef VECGEOM_USOLIDS
  if (initial_ != kBenchmarkUSolids)       delete usolids_;
#endif
}

} // End global namespace