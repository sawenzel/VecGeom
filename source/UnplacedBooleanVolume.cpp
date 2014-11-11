/*
 * UnplacedBooleanVolume.cpp
 *
 *  Created on: 07.11.2014
 *      Author: swenzel
 */

#include "base/Global.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/SpecializedBooleanVolume.h"
#include "management/VolumeFactory.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/CudaManager.h"

namespace VECGEOM_NAMESPACE
{


template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedBooleanVolume::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement)
{
    // since this is a static function, we need to get instance of UnplacedBooleanVolume first of all from logical volume
    __attribute__((unused)) const UnplacedBooleanVolume &vol
            = static_cast<const UnplacedBooleanVolume&>( *(logical_volume->unplaced_volume()) );

    if( vol.GetOp() == kSubtraction ){
   return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kSubtraction, transCodeT, rotCodeT> >(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif
    }
    else if ( vol.GetOp() == kUnion ){
        return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kUnion, transCodeT, rotCodeT> >(
        #ifdef VECGEOM_NVCC
              logical_volume, transformation, id, placement); // TODO: add bounding box?
        #else
              logical_volume, transformation, placement);
        #endif
            }
    else if ( vol.GetOp() == kIntersection ){
        return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kIntersection, transCodeT, rotCodeT> >(
        #ifdef VECGEOM_NVCC
              logical_volume, transformation, id, placement); // TODO: add bounding box?
        #else
              logical_volume, transformation, placement);
        #endif
            }
 return NULL;
    //return nullptr;
}


VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedBooleanVolume::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume>(
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
void UnplacedBooleanVolume_CopyToGpu(
     BooleanOperation op,
     VPlacedVolume const* left,
     VPlacedVolume const* right,
     VUnplacedVolume *const gpu_ptr);


// implementation of the virtual functions CopyToGpu
VUnplacedVolume* UnplacedBooleanVolume::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {

    UnplacedBooleanVolume_CopyToGpu(fOp, fLeftVolume, fRightVolume, gpu_ptr);
    CudaAssertError();

    return gpu_ptr;

}

VUnplacedVolume* UnplacedBooleanVolume::CopyToGpu() const {
  // doing an allocation
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedBooleanVolume>();

  // construct object in pre-allocated space
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedBooleanVolume_ConstructOnGpu(
    BooleanOperation op,
    VPlacedVolume const * left,
    VPlacedVolume const * right,
    VUnplacedVolume *const gpu_ptr) {

    new(gpu_ptr) vecgeom_cuda::UnplacedBooleanVolume(
        op,
        reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(left),
        reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(right));

}

void UnplacedBooleanVolume_CopyToGpu(
        BooleanOperation op,
        VPlacedVolume const* left,
        VPlacedVolume const* right,
        VUnplacedVolume *const gpu_ptr) {

    // here we have our recursion:
    // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
    // very brute force; because this might have been copied already
    // TODO: integrate this into CUDA MGR?

    // use CUDA Manager to lookup GPU pointer
    VPlacedVolume const* leftgpuptr = CudaManager::Instance().LookupPlaced(left);
    VPlacedVolume const* rightgpuptr = CudaManager::Instance().LookupPlaced(right);

    UnplacedBooleanVolume_ConstructOnGpu<<<1, 1>>>(op, leftgpuptr, rightgpuptr, gpu_ptr);
}

#endif

} // End namespace vecgeom


