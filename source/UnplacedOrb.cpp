/// \file UnplacedOrb.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/UnplacedOrb.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#include <cassert>
#include <cmath>
#endif

#include "management/VolumeFactory.h"
#include "volumes/SpecializedOrb.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb::UnplacedOrb()
  {
    //default constructor
    fR=0;
    fRTolerance =  std::max(frTolerance, epsilon * fR);
    fCubicVolume=0;
    fSurfaceArea=0;
    fRTolI = fR -  fRTolerance;
    fRTolO = fR +  fRTolerance;
    
  }

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb::UnplacedOrb(const Precision r)
  {
    fR=r;
    fRTolerance =  std::max(frTolerance, epsilon * r);
    fCubicVolume = (4 * kPi / 3) * fR * fR * fR;
    fSurfaceArea = (4 * kPi) * fR * fR;
    fRTolI = fR -  fRTolerance;
    fRTolO = fR +  fRTolerance;
  }
  
  
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::SetRadius(const Precision r)
  {
    fR=r;
    fRTolerance =  std::max(frTolerance, epsilon * r);
    fCubicVolume = (4 * kPi / 3) * fR * fR * fR;
    fSurfaceArea = (4 * kPi) * fR * fR;
    fRTolI = fR -  fRTolerance;
    fRTolO = fR +  fRTolerance;
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision UnplacedOrb::Capacity() const
  {
      return fCubicVolume;
  }
  
  
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision UnplacedOrb::SurfaceArea() const
  {
      return fSurfaceArea;
  }
  
  VECGEOM_CUDA_HEADER_BOTH  //This line is not there in UnplacedBox.cpp
  void UnplacedOrb::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
  {
    // Returns the full 3D cartesian extent of the solid.
      aMin.Set(-fR);
      aMax.Set(fR);
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::GetParametersList(int, double* aArray)const
  {
      aArray[0] = GetRadius();
  }
  
  #ifdef VECGEOM_NVCC
  Vector3D<Precision> UnplacedOrb::GetPointOnSurface() const
  {}
  #else
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> UnplacedOrb::GetPointOnSurface() const
  {
  //  generate a random number from zero to 2UUtils::kPi...
     
  Precision phi  = RNG::Instance().uniform(0., 2.* kPi);
  Precision cosphi  = std::cos(phi);
  Precision sinphi  = std::sin(phi);

  // generate a random point uniform in area
  Precision costheta = RNG::Instance().uniform(-1., 1.);
  Precision sintheta = std::sqrt(1. - (costheta*costheta));

  return Vector3D<Precision>(fR * sintheta * cosphi, fR * sintheta * sinphi, fR * costheta);
  }
  #endif
  
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::ComputeBBox() const 
  {
  
  } 
  
  VECGEOM_CUDA_HEADER_BOTH
  std::string UnplacedOrb::GetEntityType() const
  {
      return "Orb\n";
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb* UnplacedOrb::Clone() const
  {
      return new UnplacedOrb(fR);
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& UnplacedOrb::StreamInfo(std::ostream& os) const
  //Definition taken from UOrb
  {
   int oldprc = os.precision(16);
   os << "-----------------------------------------------------------\n"
   //  << "		*** Dump for solid - " << GetName() << " ***\n"
   //  << "		===================================================\n"
   
   << " Solid type: UOrb\n"
     << " Parameters: \n"

     << "		outer radius: " << fR << " mm \n"
     << "-----------------------------------------------------------\n";
   os.precision(oldprc);

   return os;
  }

  
void UnplacedOrb::Print() const {
  printf("UnplacedOrb {%.2f}",GetRadius());
}

void UnplacedOrb::Print(std::ostream &os) const {
  os << "UnplacedOrb {" << GetRadius() <<  "}";
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
    const Precision r,
    VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedOrb::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedOrb_CopyToGpu(this->GetRadius(), gpu_ptr);
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
