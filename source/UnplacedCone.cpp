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
       if(tube.rmin() <= 0) {
         if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(NonHollowCone);
         if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(NonHollowConeWithPiSector); // == M_PI ???

         if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(NonHollowConeWithSmallerThanPiSector);
         if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(NonHollowConeWithBiggerThanPiSector);
       }
       else if(tube.rmin() > 0) {
         if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(HollowCone);
         if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(HollowConeWithPiSector); // == M_PI ???
         if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(HollowConeWithSmallerThanPiSector);
         if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(HollowConeWithBiggerThanPiSector);
       }
 #endif

       RETURN_SPECIALIZATION(GenericCone);

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

