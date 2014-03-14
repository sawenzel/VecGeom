/**
 * @file volume_pointers.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "management/volume_pointers.h"
#include "volumes/placed_volume.h"

#include "TGeoShape.h"
#include "VUSolid.hh"

namespace vecgeom {

VolumePointers::VolumePointers(VPlacedVolume const *const volume)
    : specialized_(volume), initial_(kSpecialized) {
  ConvertVolume();
}

VolumePointers::VolumePointers(VolumePointers const &other)
    : specialized_(other.specialized_), initial_(kRoot) {
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
  if (!root_)          root_          = specialized_->ConvertToRoot();
  if (!usolids_)       usolids_       = specialized_->ConvertToUSolids();
}

void VolumePointers::Deallocate() {
  if (initial_ != kSpecialized)   delete specialized_;
  if (initial_ != kUnspecialized) delete unspecialized_;
  if (initial_ != kRoot)          delete root_;
  if (initial_ != kUSolids)       delete usolids_;
}

} // End namespace vecgeom