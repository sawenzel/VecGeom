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

#ifdef VECGEOM_CUDA_INTERFACE
#include "management/CudaManager.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

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
   const UnplacedBooleanVolume &vol
            = static_cast<const UnplacedBooleanVolume&>( *(logical_volume->GetUnplacedVolume()) );

   if( vol.GetOp() == kSubtraction ) {
      return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kSubtraction, transCodeT, rotCodeT> >(
      logical_volume, transformation, 
#ifdef VECGEOM_NVCC
      id,
#endif 
      placement); // TODO: add bounding box?
   }
   else if ( vol.GetOp() == kUnion ) {
      return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kUnion, transCodeT, rotCodeT> >(
              logical_volume, transformation,
        #ifdef VECGEOM_NVCC
              id,
        #endif
              placement); // TODO: add bounding box?
   }
   else if ( vol.GetOp() == kIntersection ){
      return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kIntersection, transCodeT, rotCodeT> >(
              logical_volume, transformation,
        #ifdef VECGEOM_NVCC
              id,
        #endif
              placement); // TODO: add bounding box?
   }
   return nullptr;
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

#ifndef VECGEOM_NVCC

   return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume>(
    volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
    id,
#endif
    placement);

#else
   // Compiling the above code with nvcc 6.5 faile with the error:
   // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
   // at least when optimized.
   return nullptr;
#endif
}

  void UnplacedBooleanVolume::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const {

    Vector3D<Precision> minLeft, maxLeft, minRight, maxRight;
    fLeftVolume->Extent(minLeft, maxLeft);
    fRightVolume->Extent(minRight,maxRight);

    if( this->GetOp() == kUnion ) {
      aMin = Vector3D<Precision>( Min(minLeft.x(), minRight.x()),
                                  Min(minLeft.y(), minRight.y()),
                                  Min(minLeft.z(), minRight.z()));
      aMax = Vector3D<Precision>( Max(maxLeft.x(), maxRight.x()),
                                  Max(maxLeft.y(), maxRight.y()),
                                  Max(maxLeft.z(), maxRight.z()));
    }

    if( this->GetOp() == kIntersection ) {
      aMin = Vector3D<Precision>( Max(minLeft.x(), minRight.x()),
                                  Max(minLeft.y(), minRight.y()),
                                  Max(minLeft.z(), minRight.z()));
      aMax = Vector3D<Precision>( Min(maxLeft.x(), maxRight.x()),
                                  Min(maxLeft.y(), maxRight.y()),
                                  Min(maxLeft.z(), maxRight.z()));
    }

    if( this->GetOp() == kSubtraction ) {
      aMin = minLeft;
      aMax = maxLeft;
    }
  }

#ifdef VECGEOM_CUDA_INTERFACE

// functions to copy data structures to GPU
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
    // here we have our recursion:
    // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
    // very brute force; because this might have been copied already
    // TODO: integrate this into CUDA MGR?

    // use CUDA Manager to lookup GPU pointer
    DevicePtr<cuda::VPlacedVolume> leftgpuptr = CudaManager::Instance().LookupPlaced(fLeftVolume);
    DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(fRightVolume);

    return CopyToGpuImpl<UnplacedBooleanVolume>(in_gpu_ptr, fOp, leftgpuptr, rightgpuptr);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedBooleanVolume>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedBooleanVolume>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume>::Construct(
     BooleanOperation op,
     DevicePtr<cuda::VPlacedVolume> left,
     DevicePtr<cuda::VPlacedVolume> right) const;

} // End cxx namespace

#endif


} // End namespace vecgeom


