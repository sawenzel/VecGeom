/*
 * PlacedBooleanVolume.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: swenzel
 */

#include "volumes/PlacedBooleanVolume.h"
#include "volumes/SpecializedBooleanVolume.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/LogicalVolume.h"

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE
// the actual function doing the transfer
void PlacedBooleanVolume_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedBooleanVolume::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedBooleanVolume_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedBooleanVolume::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedBooleanVolume>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

//class LogicalVolume;
class Transformation3D;
class VPlacedVolume;
class LogicalVolume;

// construction function on GPU
__global__
void PlacedBooleanVolume_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {


    vecgeom_cuda::VUnplacedVolume const * unplaced
        = (reinterpret_cast<vecgeom_cuda::LogicalVolume const *>(logical_volume))->unplaced_volume();
    vecgeom_cuda::UnplacedBooleanVolume const * unplacedboolean
        = reinterpret_cast<vecgeom_cuda::UnplacedBooleanVolume const *>(unplaced);

    BooleanOperation op = unplacedboolean->GetOp();
    if(op == kSubtraction)
    {
        new(gpu_ptr) vecgeom_cuda::GenericPlacedSubtractionVolume(
            reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
            reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
        NULL,
        id);
    }
    if(op == kUnion)
    {
        new(gpu_ptr) vecgeom_cuda::GenericPlacedUnionVolume(
            reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
            reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
           NULL,
          id);
    }
    if(op == kIntersection)
    {
        new(gpu_ptr) vecgeom_cuda::GenericPlacedIntersectionVolume(
            reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
            reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
            NULL,
            id);
    }
}

// implementation of actual copy function to cpu
// calling constructor on gpu with prealocated memory
void PlacedBooleanVolume_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {

    PlacedBooleanVolume_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
