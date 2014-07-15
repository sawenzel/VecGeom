/// \file UnplacedOrb.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/UnplacedOrb.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedOrb.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb::UnplacedOrb():dimensions_(0.0)
  {
    //default constructor
    fR=0;
    fRTolerance=0;
    fCubicVolume=0;
    fSurfaceArea=0;
    fRTolI=0;
    fRTolO=0;
    
  }

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb::UnplacedOrb(const Precision r):dimensions_(r)
  {
    SetRadius(r);
  }
  
  /*
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb::UnplacedOrb(const Precision r, const Precision rTol)
  {
    SetRadiusAndRadialTolerance(r,rTol);
  }
  */
  
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::SetRadius(const Precision r)
  {
    fR=r;
    fRTolI = fR - kHalfTolerance;
    fRTolO = fR + kHalfTolerance;
    //dimensions_(fR);
  }
  
  /*
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::SetRadialTolerance(const Precision rTol)
  {
    fRTolerance=rTol;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::SetRadiusAndRadialTolerance(const Precision r, const Precision rTol)
  {
    fR=r;
    fRTolerance=rTol;
  }
  */

void UnplacedOrb::Print() const {
  printf("UnplacedOrb {%.2f, %.2f}",GetRadius(),GetRadialTolerance());
}

void UnplacedOrb::Print(std::ostream &os) const {
  os << "UnplacedOrb {" << GetRadius() << ", "<<GetRadialTolerance() << "}";
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedOrb::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

  return CreateSpecializedWithPlacement<SpecializedOrb<transCodeT, rotCodeT> >(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif


}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedOrb::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<
      UnplacedOrb>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedOrb_CopyToGpu(
    const Precision x, const Precision y, const Precision z,
    const Precision alpha, const Precision theta, const Precision phi,
    VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedOrb::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedOrb_CopyToGpu(GetX(), GetY(), GetZ(),
                                   fAlpha, fTheta, fPhi, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedOrb::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedOrb>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedOrb_ConstructOnGpu(
    const Precision r,
    VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedOrb(r);
}

void UnplacedOrb_CopyToGpu(
    const Precision r,
    VUnplacedVolume *const gpu_ptr) {
  UnplacedOrb_ConstructOnGpu<<<1, 1>>>(r, gpu_ptr);
}

#endif

} // End namespace vecgeom
