#include "UBox.hh"
#include "TGeoBBox.h"
#include "volumes/placed_volume.h"
#include "comparison/volume_converter.h"

namespace vecgeom {

VolumeConverter::VolumeConverter(VPlacedVolume const *const volume)
    : vecgeom_(volume) {
  root_ = volume->ConvertToRoot();
  usolids_ = volume->ConvertToUSolids();
}

VolumeConverter::~VolumeConverter() {
  delete root_;
  delete usolids_;
}

} // End namespace vecgeom