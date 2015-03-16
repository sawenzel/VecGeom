/// \file UnplacedOrb.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/UnplacedOrb.h"
#include "backend/Backend.h"

#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#include <cassert>
#include <cmath>
#endif

#include "management/VolumeFactory.h"
#include "volumes/SpecializedOrb.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

   
VECGEOM_CUDA_HEADER_BOTH
UnplacedOrb::UnplacedOrb() :
   fR(0),
   fRTolerance(Max(frTolerance, epsilon * fR)),
   fRTolI(fR -  fRTolerance),
   fRTolO(fR +  fRTolerance),
   fCubicVolume(0),
   fSurfaceArea(0),
   epsilon(2e-11),
   frTolerance(1e-9)
{
    //default constructor
}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb::UnplacedOrb(const Precision r) :
   fR(r),
   fRTolerance(Max(frTolerance, epsilon * fR)),
   fRTolI(fR -  fRTolerance),
   fRTolO(fR +  fRTolerance),
   fCubicVolume((4 * kPi / 3) * fR * fR * fR),
   fSurfaceArea((4 * kPi) * fR * fR),
   epsilon(2e-11),
   frTolerance(1e-9)
  {
  }
  
  
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedOrb::SetRadius(const Precision r)
  {
    fR=r;
    fRTolerance =  Max(frTolerance, epsilon * r);
    fCubicVolume = (4 * kPi / 3) * fR * fR * fR;
    fSurfaceArea = (4 * kPi) * fR * fR;
    fRTolI = fR -  fRTolerance;
    fRTolO = fR +  fRTolerance;
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

#if !defined(VECGEOM_NVCC) && defined(VECGEOM_USOLIDS)
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
  
  //VECGEOM_CUDA_HEADER_BOTH
  std::string UnplacedOrb::GetEntityType() const
  {
      return "Orb\n";
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb* UnplacedOrb::Clone() const
  {
      return new UnplacedOrb(fR);
  }
  
  //VECGEOM_CUDA_HEADER_BOTH
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

#ifndef VECGEOM_NVCC

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedOrb::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedOrb<trans_code, rot_code>(logical_volume,
                                                        transformation);
    return placement;
  }
  return new SpecializedOrb<trans_code, rot_code>(logical_volume,
                                                  transformation);
}

VPlacedVolume* UnplacedOrb::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedOrb>(
           volume, transformation, trans_code, rot_code, placement
         );
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedOrb::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedOrb<trans_code, rot_code>(logical_volume,
                                                        transformation, NULL, id);
    return placement;
  }
  return new SpecializedOrb<trans_code, rot_code>(logical_volume,
                                                  transformation,NULL, id);
}

__device__
VPlacedVolume* UnplacedOrb::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedOrb>(
           volume, transformation, trans_code, rot_code, id, placement
         );
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedOrb::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedOrb>(in_gpu_ptr, GetRadius());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedOrb::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedOrb>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedOrb>::SizeOf();
template void DevicePtr<cuda::UnplacedOrb>::Construct(const Precision r) const;

} // End cxx namespace

#endif

} // End global namespace
