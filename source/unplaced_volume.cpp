/**
 * @file unplaced_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "volumes/unplaced_volume.h"

namespace vecgeom {

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol) {
  vol.Print(os);
  return os;
}

VPlacedVolume* VUnplacedVolume::PlaceVolume(
    LogicalVolume const *const volume,
    TransformationMatrix const *const matrix) const {

  const TranslationCode trans_code = matrix->GenerateTranslationCode();
  const RotationCode rot_code = matrix->GenerateRotationCode();

  return SpecializedVolume(volume, matrix, trans_code, rot_code);
}

} // End namespace vecgeom