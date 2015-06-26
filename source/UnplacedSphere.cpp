/// \file UnplacedSphere.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/UnplacedSphere.h"
#include "backend/Backend.h"

#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#include <cassert>
#include <cmath>
#endif

#include "management/VolumeFactory.h"
#include "volumes/SpecializedSphere.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "volumes/SphereUtilities.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


  VECGEOM_CUDA_HEADER_BOTH
  UnplacedSphere::UnplacedSphere(Precision pRmin, Precision pRmax,
                 Precision pSPhi, Precision pDPhi,

                 Precision pSTheta, Precision pDTheta)  :
     fRmax(0),
     fSPhi(0),
     fDPhi(0),
     fSTheta(0),
     fDTheta(0),
     fRminTolerance(0),
     mkTolerance(0),
     fEpsilon(kEpsilon), 
     sinCPhi(0),
     cosCPhi(0),
     cosHDPhiOT(0),
     cosHDPhiIT(0),
     sinSPhi(0),
     cosSPhi(0),
     sinEPhi(0),
     cosEPhi(0),
     hDPhi(0),
     cPhi(0),
     ePhi(0),
     sinSTheta(0),
     cosSTheta(0),
     sinETheta(0),
     cosETheta(0),
     tanSTheta(0),
     tanSTheta2(0),
     tanETheta(0),
     tanETheta2(0),
     eTheta(0),
     fFullPhiSphere(true), 
     fFullThetaSphere(true),
     fFullSphere(true),
     fCubicVolume(0.), 
     fSurfaceArea(0.), 
     //epsilon(kSEpsilon), 
     // frTolerance(kTolerance*10.),
     fPhiWedge(pDPhi,pSPhi),
     fThetaCone(pSTheta,pDTheta)
            
{
  // Check radii and Set radial tolerances

 
  if ((pRmin >= pRmax) || (pRmax < 1.1 * kRadTolerance) || (pRmin < 0))
  {/*
    std::ostringstream message;
    message << "Invalid radii for Solid: " ;//<< GetName() << std::endl
            << std::endl<<"             pRmin = " << pRmin << ", pRmax = " << pRmax;
                return;
    */ 
    //UUtils::Exception("USphere::USphere()", "GeomSolids0002",
     //                 FatalErrorInArguments, 1, message.str().c_str());
  }
  fRmin = pRmin;
  fRmax = pRmax;
  fRminTolerance = (fRmin) ? Max(kRadTolerance, fEpsilon * fRmin) : 0;
  mkTolerance = Max(kRadTolerance, fEpsilon * fRmax);//RELOOK //kTolerance is replaced by mkTolerance

  // Check angles

  CheckPhiAngles(pSPhi, pDPhi);
  CheckThetaAngles(pSTheta, pDTheta);
  
#ifndef VECGEOM_NVCC
  CalcCapacity();
  CalcSurfaceArea();
#endif
}
  
#ifndef VECGEOM_NVCC
  void UnplacedSphere::CalcCapacity()
  {
      if (fCubicVolume != 0.)
        {
         ;
        }
      else
        {
             fCubicVolume = fDPhi * (std::cos(fSTheta) - std::cos(fSTheta + fDTheta)) *
                (fRmax * fRmax * fRmax - fRmin * fRmin * fRmin) / 3.;
        }
      
  }
  

  void UnplacedSphere::CalcSurfaceArea()
  {
      
      if (fSurfaceArea != 0.)
        {
          ;
        }
      else
         {
            Precision Rsq = fRmax * fRmax;
            Precision rsq = fRmin * fRmin;

            fSurfaceArea = fDPhi * (rsq + Rsq) * (cosSTheta - cosETheta);
            if (!fFullPhiSphere)
            {
              fSurfaceArea = fSurfaceArea + fDTheta * (Rsq - rsq);
            }
            if (fSTheta > 0)
            {
             Precision acos1 = std::acos(Pow(sinSTheta, 2) * std::cos(fDPhi)
                               + Pow(cosSTheta, 2));
            if (fDPhi > kPi)
            {
              fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * (2 * kPi - acos1);
            }
             else
             {
                fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * acos1;
             }
            }
            if (eTheta < kPi)
            {
             double acos2 = std::acos(Pow(sinETheta, 2) * std::cos(fDPhi)
                               + Pow(cosETheta, 2));
            if (fDPhi > kPi)
            {
              fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * (2 * kPi - acos2);
            }
            else
            {
              fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * acos2;
            }
            }
        }
  }
  
  
  void UnplacedSphere::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
  {
    // Returns the full 3D cartesian extent of the solid.
      aMin.Set(-fRmax);
      aMax.Set(fRmax);
  }
#endif // !VECGEOM_NVCC

  void UnplacedSphere::GetParametersList(int, double* aArray)const
  {
      aArray[0] = GetInnerRadius();
      aArray[1] = GetOuterRadius();
      aArray[2] = GetStartPhiAngle();
      aArray[3] = GetDeltaPhiAngle();
      aArray[4] = GetStartThetaAngle();
      aArray[5] = GetDeltaThetaAngle();
  }
  
  
  #ifndef VECGEOM_NVCC
  Vector3D<Precision> UnplacedSphere::GetPointOnSurface() const
  {
      
  Precision zRand, aOne, aTwo, aThr, aFou, aFiv, chose, phi, sinphi, cosphi;
  Precision height1, height2, slant1, slant2, costheta, sintheta, rRand;

  height1 = (fRmax - fRmin) * cosSTheta;
  height2 = (fRmax - fRmin) * cosETheta;
  slant1  = std::sqrt(sqr((fRmax - fRmin) * sinSTheta) + height1 * height1);
  slant2  = std::sqrt(sqr((fRmax - fRmin) * sinETheta) + height2 * height2);
  rRand  = GetRadiusInRing(fRmin, fRmax);

  aOne = fRmax * fRmax * fDPhi * (cosSTheta - cosETheta);
  aTwo = fRmin * fRmin * fDPhi * (cosSTheta - cosETheta);
  aThr = fDPhi * ((fRmax + fRmin) * sinSTheta) * slant1;
  aFou = fDPhi * ((fRmax + fRmin) * sinETheta) * slant2;
  aFiv = 0.5 * fDTheta * (fRmax * fRmax - fRmin * fRmin);

  phi = RNG::Instance().uniform(fSPhi, ePhi);
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  costheta = RNG::Instance().uniform(cosETheta, cosSTheta);
  sintheta = std::sqrt(1. - sqr(costheta));

  if (fFullPhiSphere)
  {
    aFiv = 0;
  }
  if (fSTheta == 0)
  {
    aThr = 0;
  }
  if (eTheta == kPi)
  {
    aFou = 0;
  }
  if (fSTheta == kPi / 2)
  {
    aThr = kPi * (fRmax * fRmax - fRmin * fRmin);
  }
  if (eTheta == kPi / 2)
  {
    aFou = kPi * (fRmax * fRmax - fRmin * fRmin);
  }

  chose = RNG::Instance().uniform(0., aOne + aTwo + aThr + aFou + 2.*aFiv);
  if ((chose >= 0.) && (chose < aOne))
  {
    return Vector3D<Precision>(fRmax * sintheta * cosphi,
                    fRmax * sintheta * sinphi, fRmax * costheta);
  }
  else if ((chose >= aOne) && (chose < aOne + aTwo))
  {
    return Vector3D<Precision>(fRmin * sintheta * cosphi,
                    fRmin * sintheta * sinphi, fRmin * costheta);
  }
  else if ((chose >= aOne + aTwo) && (chose < aOne + aTwo + aThr))
  {
    if (fSTheta != kPi / 2)
    {
      zRand = RNG::Instance().uniform(fRmin * cosSTheta, fRmax * cosSTheta);
      return Vector3D<Precision>(tanSTheta * zRand * cosphi,
                      tanSTheta * zRand * sinphi, zRand);
    }
    else
    {
      return Vector3D<Precision>(rRand * cosphi, rRand * sinphi, 0.);
    }
  }
  else if ((chose >= aOne + aTwo + aThr) && (chose < aOne + aTwo + aThr + aFou))
  {
    if (eTheta != kPi / 2)
    {
      zRand = RNG::Instance().uniform(fRmin * cosETheta, fRmax * cosETheta);
      return Vector3D<Precision>(tanETheta * zRand * cosphi,
                      tanETheta * zRand * sinphi, zRand);
    }
    else
    {
      return Vector3D<Precision>(rRand * cosphi, rRand * sinphi, 0.);
    }
  }
  else if ((chose >= aOne + aTwo + aThr + aFou) && (chose < aOne + aTwo + aThr + aFou + aFiv))
  {
    return Vector3D<Precision>(rRand * sintheta * cosSPhi,
                    rRand * sintheta * sinSPhi, rRand * costheta);
  }
  else
  {
    return Vector3D<Precision>(rRand * sintheta * cosEPhi,
                    rRand * sintheta * sinEPhi, rRand * costheta);
  }
  
  }


  std::string UnplacedSphere::GetEntityType() const
  {
      return "Sphere\n";
  }

#endif // !VECGEOM_NVCC
  
  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedSphere::ComputeBBox() const 
  {
  
  } 
  
  UnplacedSphere* UnplacedSphere::Clone() const
  {
      return new UnplacedSphere(fRmin,fRmax,fSPhi,fDPhi,fSTheta,fDTheta);
  }
  
#if defined(VECGEOM_USOLIDS)
  std::ostream& UnplacedSphere::StreamInfo(std::ostream& os) const
  //Definition taken from USphere
  {
      
   int oldprc = os.precision(16);
   os << "-----------------------------------------------------------\n"
   //  << "     *** Dump for solid - " << GetName() << " ***\n"
   //  << "     ===================================================\n"
   
   << " Solid type: VecGeomSphere\n"
     << " Parameters: \n"

     << "       outer radius: " << fRmax << " mm \n"
     << "               Inner radius: " <<fRmin <<"mm\n"    
     << "               Start Phi Angle: "<<fSPhi<<"\n"
     << "               Delta Phi Angle: "<<fDPhi<<"\n"
     << "               Start Theta Angle: "<<fSTheta<<"\n"
     << "               Delta Theta Angle: "<<fDTheta<<"\n"
     << "-----------------------------------------------------------\n";
   os.precision(oldprc);

   return os;
  }
#endif
  
  VECGEOM_CUDA_HEADER_BOTH
void UnplacedSphere::Print() const {
  printf("UnplacedSphere {%.2f , %.2f , %.2f , %.2f , %.2f , %.2f}",GetInnerRadius(),GetOuterRadius(),
                                                          GetStartPhiAngle(), GetDeltaPhiAngle(),
                                                          GetStartThetaAngle(), GetDeltaThetaAngle() );
}
  
VECGEOM_CUDA_HEADER_BOTH
void UnplacedSphere::Print(std::ostream &os) const {
  os << "UnplacedSphere { " << GetInnerRadius() <<" " << GetOuterRadius() <<" " << GetStartPhiAngle() <<" " << GetDeltaPhiAngle() <<" "
           << GetStartThetaAngle() <<" " << GetDeltaThetaAngle() <<" }";
}

  
#ifndef VECGEOM_NVCC

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedSphere::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedSphere<trans_code, rot_code>(logical_volume,
                                                        transformation);
    return placement;
  }
  return new SpecializedSphere<trans_code, rot_code>(logical_volume,
                                                  transformation);
}

VPlacedVolume* UnplacedSphere::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedSphere>(
           volume, transformation, trans_code, rot_code, placement
         );
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedSphere::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedSphere<trans_code, rot_code>(logical_volume,
                                                        transformation, NULL, id);
    return placement;
  }
  return new SpecializedSphere<trans_code, rot_code>(logical_volume,
                                                  transformation,NULL, id);
}

__device__
VPlacedVolume* UnplacedSphere::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedSphere>(
           volume, transformation, trans_code, rot_code, id, placement
         );
}

#endif


#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedSphere::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedSphere>(in_gpu_ptr, GetInnerRadius(), GetOuterRadius(), GetStartPhiAngle(),
                                        GetDeltaPhiAngle(), GetStartThetaAngle(),GetDeltaThetaAngle());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedSphere::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedSphere>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedSphere>::SizeOf();
template void DevicePtr<cuda::UnplacedSphere>::Construct(
    const Precision rmin, const Precision rmax, const Precision sphi,
    const Precision dphi, const Precision stheta,const Precision dtheta) const;

} // End cxx namespace

#endif

} // End global namespace

