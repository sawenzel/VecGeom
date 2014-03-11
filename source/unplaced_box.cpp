#include <stdio.h>
#include "backend.h"
#include "management/volume_factory.h"
#include "volumes/specialized_box.h"
#include "volumes/unplaced_box.h"

namespace vecgeom {

#ifdef VECGEOM_CUDA

namespace {

__global__
void ConstructOnGpu(const UnplacedBox box,
                    VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) UnplacedBox(box);
}

} // End anonymous namespace

VUnplacedVolume* UnplacedBox::CopyToGpu(VUnplacedVolume *const gpu_ptr) const {
  ConstructOnGpu<<<1, 1>>>(*this, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedBox::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedBox>();
  return CopyToGpu(gpu_ptr);
}

#endif

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedBox::Create(
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix) {
  return new SpecializedBox<trans_code, rot_code>(logical_volume, matrix);
}

VPlacedVolume* UnplacedBox::SpecializedVolume(
    LogicalVolume const *const volume,
    TransformationMatrix const *const matrix,
    const TranslationCode trans_code, const RotationCode rot_code) const {
  return VolumeFactory::Instance().CreateByTransformation<UnplacedBox>(
           volume, matrix, trans_code, rot_code
         );
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedBox::Print() const {
  printf("Box {%f, %f, %f}", dimensions_[0], dimensions_[1], dimensions_[2]);
}

} // End namespace vecgeom