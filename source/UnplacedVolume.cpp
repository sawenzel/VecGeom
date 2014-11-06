/// \file UnplacedVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedVolume.h"

#include "volumes/PlacedVolume.h"

namespace VECGEOM_NAMESPACE {

std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol) {
  vol.Print(os);
  return os;
}

#ifndef VECGEOM_NVCC

VPlacedVolume* VUnplacedVolume::PlaceVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) const {

  const TranslationCode trans_code = transformation->GenerateTranslationCode();
  const RotationCode rot_code = transformation->GenerateRotationCode();

  return SpecializedVolume(volume, transformation, trans_code, rot_code,
                           placement);
}

VPlacedVolume* VUnplacedVolume::PlaceVolume(
    char const *const label,
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) const {
  VPlacedVolume *const placed = PlaceVolume(volume, transformation, placement);
  // placed->set_label(label);
  return placed;
}

#endif

} // End global namespace
