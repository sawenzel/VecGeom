/// @file UnplacedParaboloid.cpp

#include "volumes/UnplacedParaboloid.h"

#include "management/volume_factory.h"
#include "volumes/SpecializedParaboloid.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

void UnplacedParaboloid::Print() const {
  // NYI
}

void UnplacedParaboloid::Print(std::ostream &os) const {
  // NYI
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParaboloid::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  if (placement) {
    return new(placement) SpecializedParaboloid<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
        logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
        logical_volume, transformation);
#endif
  }
  return new SpecializedParaboloid<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParaboloid::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<
      UnplacedParaboloid>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedParaboloid_CopyToGpu(VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedParaboloid::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedParaboloid_CopyToGpu(gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedParaboloid::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedParaboloid>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedParaboloid_ConstructOnGpu(VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedParaboloid();
}

void UnplacedParaboloid_CopyToGpu(VUnplacedVolume *const gpu_ptr) {
  UnplacedParaboloid_ConstructOnGpu<<<1, 1>>>(gpu_ptr);
}

#endif

} // End namespace vecgeom