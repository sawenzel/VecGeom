/*
 * UnplacedCone.cpp
 *
 *  Created on: Jun 18, 2014
 *      Author: swenzel
 */


#include "volumes/UnplacedCone.h"
#include "volumes/SpecializedCone.h"
#include "volumes/utilities/GenerationUtilities.h"

#include "management/VolumeFactory.h"

namespace VECGEOM_NAMESPACE {

    void UnplacedCone::Print() const {
     printf("UnplacedCone {rmin1 %.2f, rmax1 %.2f, rmin2 %.2f, "
          "rmax2 %.2f, phistart %.2f, deltaphi %.2f}",
             fRmin1, fRmax2, fRmin2, fRmax2, fSPhi, fDPhi);
    }

    void UnplacedCone::Print(std::ostream &os) const {
        os << "UnplacedCone; please implement Print to outstream\n";
    }

    // what else to implement ??

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  VPlacedVolume* UnplacedCone::Create(
     LogicalVolume const *const logical_volume,
     Transformation3D const *const transformation,
 #ifdef VECGEOM_NVCC
     const int id,
 #endif
     VPlacedVolume *const placement) {

       using namespace ConeTypes;
       __attribute__((unused)) const UnplacedCone &cone = static_cast<const UnplacedCone&>( *(logical_volume->unplaced_volume()) );

       #ifdef VECGEOM_NVCC
         #define RETURN_SPECIALIZATION(coneTypeT) return CreateSpecializedWithPlacement< \
             SpecializedCone<transCodeT, rotCodeT, coneTypeT> >(logical_volume, transformation, id, placement)
       #else
         #define RETURN_SPECIALIZATION(coneTypeT) return CreateSpecializedWithPlacement< \
             SpecializedCone<transCodeT, rotCodeT, coneTypeT> >(logical_volume, transformation, placement)
       #endif

  #ifdef GENERATE_CONE_SPECIALIZATIONS
       if(cone.GetRmin1() <= 0 && cone.GetRmin2() <=0) {
         if(cone.GetDPhi() >= 2*M_PI)  RETURN_SPECIALIZATION(NonHollowCone);
         if(cone.GetDPhi() == M_PI)    RETURN_SPECIALIZATION(NonHollowConeWithPiSector); // == M_PI ???

         if(cone.GetDPhi() < M_PI)     RETURN_SPECIALIZATION(NonHollowConeWithSmallerThanPiSector);
         if(cone.GetDPhi() > M_PI)     RETURN_SPECIALIZATION(NonHollowConeWithBiggerThanPiSector);
       }
       else if(cone.GetRmin1() > 0 || cone.GetRmin2() > 0) {
         if(cone.GetDPhi() >= 2*M_PI)  RETURN_SPECIALIZATION(HollowCone);
         if(cone.GetDPhi() == M_PI)    RETURN_SPECIALIZATION(HollowConeWithPiSector); // == M_PI ???
         if(cone.GetDPhi() < M_PI)     RETURN_SPECIALIZATION(HollowConeWithSmallerThanPiSector);
         if(cone.GetDPhi() > M_PI)     RETURN_SPECIALIZATION(HollowConeWithBiggerThanPiSector);
       }
 #endif

       RETURN_SPECIALIZATION(SimpleCone);

       #undef RETURN_SPECIALIZATION
 }


// this is repetetive code:

  VECGEOM_CUDA_HEADER_DEVICE
  VPlacedVolume* UnplacedCone::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedCone>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}


}



namespace vecgeom {


#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedCone_CopyToGpu(
    const Precision rmin1, const Precision rmax1,
    const Precision rmin2, const Precision rmax2,
    const Precision z, const Precision sphi, const Precision dphi,
    VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedCone::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedCone_CopyToGpu(GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetDz(), GetSPhi(), GetDPhi(), gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedCone::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedCone>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedCone_ConstructOnGpu(
    const Precision rmin1, const Precision rmax1,
    const Precision rmin2, const Precision rmax2,
    const Precision z, const Precision sphi, const Precision dphi,
    VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedCone(rmin1, rmax1, rmin2, rmax2, z, sphi, dphi);
}

void UnplacedCone_CopyToGpu(
        const Precision rmin1, const Precision rmax1,
        const Precision rmin2, const Precision rmax2,
        const Precision z, const Precision sphi, const Precision dphi,
        VUnplacedVolume *const gpu_ptr) {
  UnplacedCone_ConstructOnGpu<<<1, 1>>>(rmin1, rmax1, rmin2, rmax2, z, sphi, dphi, gpu_ptr);
}

#endif

} // End namespace vecgeom



