/// @file unplaced_root_volume.cpp
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/unplaced_root_volume.h"

#include "volumes/placed_root_volume.h"

#include <stdio.h>

namespace vecgeom {

void UnplacedRootVolume::Print() const {
  printf("UnplacedRootVolume");
}

void UnplacedRootVolume::Print(std::ostream &os) const {
  os << "UnplacedRootVolume";
}

VPlacedVolume* UnplacedRootVolume::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) const {
  if (placement) {
    return new(placement) PlacedRootVolume(fRootShape, volume, transformation);
  }
  return new PlacedRootVolume(fRootShape, volume, transformation);
}

#ifdef VECGEOM_CUDA_INTERFACE
VUnplacedVolume* UnplacedRootVolume::CopyToGpu() const {
  assert(0 && "Attempted to copy unsupported ROOT volume to GPU.");
}
VUnplacedVolume* UnplacedRootVolume::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  assert(0 && "Attempted to copy unsupported ROOT volume to GPU.");
}
#endif

} // End namespace vecgeom