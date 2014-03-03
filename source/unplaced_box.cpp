#include <stdio.h>
#include "volumes/unplaced_box.h"
#ifdef VECGEOM_NVCC
#include "backend/cuda_backend.cuh"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

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

VECGEOM_CUDA_HEADER_BOTH
void UnplacedBox::Print() const {
  printf("Box {%f, %f, %f}", dimensions_[0], dimensions_[1], dimensions_[2]);
}

} // End namespace vecgeom