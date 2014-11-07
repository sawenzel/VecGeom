/*
 * UnplacedBooleanMinusVolume.cpp
 *
 *  Created on: 07.11.2014
 *      Author: swenzel
 */

#include "base/Global.h"
#include "volumes/UnplacedBooleanMinusVolume.h"
#include "volumes/SpecializedBooleanMinusVolume.h"
#include "management/VolumeFactory.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/CudaManager.h"

namespace VECGEOM_NAMESPACE
{


template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedBooleanMinusVolume::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  return CreateSpecializedWithPlacement<SpecializedBooleanMinusVolume<transCodeT, rotCodeT> >(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedBooleanMinusVolume::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<UnplacedBooleanMinusVolume>(
    volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
    id,
#endif
    placement);
}


}


// functions to copy data structures to GPU
namespace vecgeom {


#ifdef VECGEOM_CUDA_INTERFACE

// declaration of a common helper function
void UnplacedBooleanMinusVolume_CopyToGpu(
     VPlacedVolume const* left,
     VPlacedVolume const* right,
     VUnplacedVolume *const gpu_ptr);


// implementation of the virtual functions CopyToGpu
VUnplacedVolume* UnplacedBooleanMinusVolume::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {

    UnplacedBooleanMinusVolume_CopyToGpu(fLeftVolume, fRightVolume, gpu_ptr);
    CudaAssertError();

    return gpu_ptr;

}

VUnplacedVolume* UnplacedBooleanMinusVolume::CopyToGpu() const {
  // doing an allocation
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedBooleanMinusVolume>();

  // construct object in pre-allocated space
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedBooleanMinusVolume_ConstructOnGpu(
    VPlacedVolume const * left,
    VPlacedVolume const * right,
    VUnplacedVolume *const gpu_ptr) {

    new(gpu_ptr) vecgeom_cuda::UnplacedBooleanMinusVolume(
        reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(left),
        reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(right));

}

void UnplacedBooleanMinusVolume_CopyToGpu(
        VPlacedVolume const* left,
        VPlacedVolume const* right,
        VUnplacedVolume *const gpu_ptr) {

    // here we have our recursion:
    // since UnplacedBooleanMinusVolume has pointer members we need to copy/construct those members too
    // very brute force; because this might have been copied already
    // TODO: integrate this into CUDA MGR?

    // use CUDA Manager to lookup GPU pointer
    VPlacedVolume const* leftgpuptr = CudaManager::Instance().LookupPlaced(left);
    VPlacedVolume const* rightgpuptr = CudaManager::Instance().LookupPlaced(right);

    UnplacedBooleanMinusVolume_ConstructOnGpu<<<1, 1>>>(leftgpuptr, rightgpuptr, gpu_ptr);
}

#endif

} // End namespace vecgeom


