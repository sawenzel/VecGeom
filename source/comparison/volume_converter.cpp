#include "UBox.hh"
#include "TGeoBBox.h"
#include "volumes/placed_volume.h"
#include "comparison/volume_converter.h"

namespace vecgeom {

VolumeConverter::VolumeConverter(VPlacedVolume const *const volume)
    : specialized_(volume) {
  ConvertVolume();
}

VolumeConverter::VolumeConverter(VolumeConverter const &other)
    : specialized_(other.specialized_) {
  ConvertVolume();
}

VolumeConverter::~VolumeConverter() {
  Deallocate();
}

VolumeConverter& VolumeConverter::operator=(VolumeConverter const &other) {
  this->Deallocate();
  this->specialized_ = other.specialized_;
  this->ConvertVolume();
  return *this;
}

void VolumeConverter::ConvertVolume() {
  unspecialized_ = specialized_->ConvertToUnspecialized();
  root_ = specialized_->ConvertToRoot();
  usolids_ = specialized_->ConvertToUSolids();
}

void VolumeConverter::Deallocate() {
  delete unspecialized_;
  delete root_;
  delete usolids_;
}

} // End namespace vecgeom