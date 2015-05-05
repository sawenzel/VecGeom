/*
 * UnplacedCone.cpp
 *
 *  Created on: Jun 18, 2014
 *      Author: swenzel
 */

#include "volumes/UnplacedCone.h"
#include "volumes/SpecializedCone.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "volumes/utilities/GenerationUtilities.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#endif
#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

    void UnplacedCone::Print() const {
     printf("UnplacedCone {rmin1 %.2f, rmax1 %.2f, rmin2 %.2f, "
          "rmax2 %.2f, dz %.2f, phistart %.2f, deltaphi %.2f}",
             fRmin1, fRmax1, fRmin2, fRmax2, fDz, fSPhi, fDPhi);
    }

    void UnplacedCone::Print(std::ostream &os) const {
        os << "UnplacedCone; please implement Print to outstream\n";
    }

#if !defined(VECGEOM_NVCC)
    Vector3D<Precision> UnplacedCone::GetPointOnSurface() const {
       // implementation taken from UCons; not verified
       //
       double Aone, Atwo, Athree, Afour, Afive, slin, slout, phi;
       double zRand, cosu, sinu, rRand1, rRand2, chose, rone, rtwo, qone, qtwo;
       rone = (fRmax1 - fRmax2) / (2.*fDz);
       rtwo = (fRmin1 - fRmin2) / (2.*fDz);
       qone = 0.;
       qtwo = 0.;
       if (fRmax1 != fRmax2){
            qone = fDz * (fRmax1 + fRmax2) / (fRmax1 - fRmax2);
       }
       if (fRmin1 != fRmin2) {
            qtwo = fDz * (fRmin1 + fRmin2) / (fRmin1 - fRmin2);
       }
       slin   = Sqrt((fRmin1 - fRmin2)*(fRmin1 - fRmin2) + 4.*fDz*fDz);
       slout = Sqrt((fRmax1 - fRmax2)*(fRmax1 - fRmax2) + 4.*fDz*fDz);
       Aone   = 0.5 * fDPhi * (fRmax2 + fRmax1) * slout;
       Atwo   = 0.5 * fDPhi * (fRmin2 + fRmin1) * slin;
       Athree = 0.5 * fDPhi * (fRmax1 * fRmax1 - fRmin1 * fRmin1);
       Afour = 0.5 * fDPhi * (fRmax2 * fRmax2 - fRmin2 * fRmin2);
       Afive = fDz * (fRmax1 - fRmin1 + fRmax2 - fRmin2);

       phi  = RNG::Instance().uniform(fSPhi, fSPhi + fDPhi);
       cosu = std::cos(phi);
       sinu = std::sin(phi);
       rRand1 = volumeUtilities::GetRadiusInRing(fRmin1, fRmin2);
       rRand2 = volumeUtilities::GetRadiusInRing(fRmax1, fRmax2);

       if ((fSPhi == 0.) && IsFullPhi()) {
         Afive = 0.;
       }
       chose = RNG::Instance().uniform(0., Aone + Atwo + Athree + Afour + 2.*Afive);

            if ((chose >= 0.) && (chose < Aone)) {
              if (fRmin1 != fRmin2)
              {
                zRand = RNG::Instance().uniform(-1.*fDz, fDz);
                return Vector3D<Precision>(rtwo * cosu * (qtwo - zRand),
                                rtwo * sinu * (qtwo - zRand), zRand);
              }
              else {
                return Vector3D<Precision>(fRmin1 * cosu, fRmin2 * sinu,
                                RNG::Instance().uniform(-1.*fDz, fDz));
              }
            }
            else if ((chose >= Aone) && (chose <= Aone + Atwo))
            {
              if (fRmax1 != fRmax2)
              {
                zRand = RNG::Instance().uniform(-1.*fDz, fDz);
                return Vector3D<Precision>(rone * cosu * (qone - zRand),
                                rone * sinu * (qone - zRand), zRand);
              }
              else
              {
                return Vector3D<Precision>(fRmax1 * cosu, fRmax2 * sinu,
                                RNG::Instance().uniform(-1.*fDz, fDz));
              }
            }
            else if ((chose >= Aone + Atwo) && (chose < Aone + Atwo + Athree))
            {
              return Vector3D<Precision>(rRand1 * cosu, rRand1 * sinu, -1 * fDz);
            }
            else if ((chose >= Aone + Atwo + Athree)
                     && (chose < Aone + Atwo + Athree + Afour))
            {
              return Vector3D<Precision>(rRand2 * cosu, rRand2 * sinu, fDz);
            }
            else if ((chose >= Aone + Atwo + Athree + Afour)
                     && (chose < Aone + Atwo + Athree + Afour + Afive))
            {
              zRand = RNG::Instance().uniform(-1.*fDz, fDz);
              rRand1 = RNG::Instance().uniform(fRmin2 - ((zRand - fDz) / (2.*fDz)) * (fRmin1 - fRmin2),
                                      fRmax2 - ((zRand - fDz) / (2.*fDz)) * (fRmax1 - fRmax2));
              return Vector3D<Precision>(rRand1 * std::cos(fSPhi),
                              rRand1 * std::sin(fSPhi), zRand);
            }
            else
            {
              zRand = RNG::Instance().uniform(-1.*fDz, fDz);
              rRand1 = RNG::Instance().uniform(fRmin2 - ((zRand - fDz) / (2.*fDz)) * (fRmin1 - fRmin2),
                                      fRmax2 - ((zRand - fDz) / (2.*fDz)) * (fRmax1 - fRmax2));
              return Vector3D<Precision>(rRand1 * std::cos(fSPhi + fDPhi),
                              rRand1 * std::sin(fSPhi + fDPhi), zRand);
            }
   }
#endif // VECGEOM_NVCC


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

       RETURN_SPECIALIZATION(UniversalCone);

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

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedCone::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedCone>(in_gpu_ptr, GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetDz(), GetSPhi(), GetDPhi());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedCone::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedCone>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedCone>::SizeOf();
template void DevicePtr<cuda::UnplacedCone>::Construct(
    const Precision rmin1, const Precision rmax1,
    const Precision rmin2, const Precision rmax2,
    const Precision z, const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End namespace vecgeom
