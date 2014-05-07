/**
 * @file unplaced_box.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "volumes/unplaced_box.h"

#include "backend/backend.h"
#include "management/volume_factory.h"
#include "volumes/specialized_box.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void UnplacedBox::Print() const {
  printf("UnplacedBox {%.2f, %.2f, %.2f}", x(), y(), z());
}

void UnplacedBox::Print(std::ostream &os) const {
  os << "UnplacedBox {" << x() << ", " << y() << ", " << z() << "}";
}

#ifndef VECGEOM_NVCC

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedBox::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedBox<trans_code, rot_code>(logical_volume,
                                                        transformation);
    return placement;
  }
  return new SpecializedBox<trans_code, rot_code>(logical_volume,
                                                  transformation);
}

VPlacedVolume* UnplacedBox::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedBox>(
           volume, transformation, trans_code, rot_code, placement
         );
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedBox::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedBox<trans_code, rot_code>(logical_volume,
                                                        transformation, id);
    return placement;
  }
  return new SpecializedBox<trans_code, rot_code>(logical_volume,
                                                  transformation, id);
}

__device__
VPlacedVolume* UnplacedBox::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedBox>(
           volume, transformation, trans_code, rot_code, id, placement
         );
}

#endif

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedBox_CopyToGpu(const Precision x, const Precision y,
                           const Precision z, VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedBox::CopyToGpu(VUnplacedVolume *const gpu_ptr) const {
  UnplacedBox_CopyToGpu(this->x(), this->y(), this->z(), gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedBox::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<UnplacedBox>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedBox_ConstructOnGpu(const Precision x, const Precision y,
                                const Precision z, VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedBox(x, y, z);
}

void UnplacedBox_CopyToGpu(const Precision x, const Precision y,
                           const Precision z, VUnplacedVolume *const gpu_ptr) {
  UnplacedBox_ConstructOnGpu<<<1, 1>>>(x, y, z, gpu_ptr);
}

#endif

} // End global namespace