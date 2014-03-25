/**
 * @file unplaced_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol) {
  vol.Print(os);
  return os;
}

#ifndef VECGEOM_NVCC
VPlacedVolume* VUnplacedVolume::PlaceVolume(
    LogicalVolume const *const volume,
    TransformationMatrix const *const matrix,
    VPlacedVolume *const placement) const {

  const TranslationCode trans_code = matrix->GenerateTranslationCode();
  const RotationCode rot_code = matrix->GenerateRotationCode();

  return SpecializedVolume(volume, matrix, trans_code, rot_code, placement);
}
#endif

} // End global namespace