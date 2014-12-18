/*
 * UnplacedPolycone.cpp
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#include "volumes/UnplacedPolycone.h"
#include "volumes/UnplacedCone.h"
#include "volumes/SpecializedPolycone.h"
#include "management/VolumeFactory.h"
#ifndef VECGEOM_NVCC
#include "base/RNG.h"
#endif
#include <iostream>
#include <cstdio>
#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {



//
// Constructor (GEANT3 style parameters)
//
void UnplacedPolycone::Init(double phiStart,
                     double phiTotal,
                     int numZPlanes,
                     const double zPlane[],
                     const double rInner[],
                     const double rOuter[])
{
  //Conversion for angles
  if (phiTotal <= 0. || phiTotal > kTwoPi-kTolerance)
   {
     // phiIsOpen=false;
     fStartPhi = 0;
     fEndPhi = kTwoPi;
   }
   else
   {
     //
     // Convert phi into our convention
     //
       // phiIsOpen=true;
     fStartPhi = phiStart;
     while( fStartPhi < 0 ) fStartPhi += kTwoPi;

     fEndPhi = fStartPhi+fDeltaPhi;
     while( fEndPhi < fStartPhi ) fEndPhi += kTwoPi;
  }

  // Calculate RMax of Polycone in order to determine convexity of sections
  //
  double RMaxextent=rOuter[0];
  for (int j=1; j < numZPlanes; j++)
  {
    if (rOuter[j] > RMaxextent) RMaxextent=rOuter[j];

    if (rInner[j] > rOuter[j])
    {
#ifndef VECGEOM_NVCC
      std::cerr << "Cannot create Polycone with rInner > rOuter for the same Z"
              << "\n"
              << "        rInner > rOuter for the same Z !\n"
              << "        rMin[" << j << "] = " << rInner[j]
              << " -- rMax[" << j << "] = " << rOuter[j];
      // UUtils::Exception("UPolycone::UPolycone()", "GeomSolids0002",
        //                 FatalErrorInArguments, 1, message.str().c_str());
#endif
    }
  }

  //
  double prevZ =  zPlane[0], prevRmax = 0, prevRmin = 0;
  int dirZ = 1;
  if (zPlane[1] < zPlane[0]) dirZ = -1;
//  int curSolid = 0;

  for (int i = 0; i < numZPlanes; ++i)
  {
    if ((i < numZPlanes - 1) && (zPlane[i] == zPlane[i + 1]))
    {
      if ((rInner[i]  > rOuter[i + 1]) || (rInner[i + 1] > rOuter[i]))
            {
#ifndef VECGEOM_NVCC
          std::cerr << "Cannot create a Polycone with no contiguous segments."
                << std::endl
                << "                Segments are not contiguous !" << std::endl
                << "                rMin[" << i << "] = " << rInner[i]
                << " -- rMax[" << i + 1 << "] = " << rOuter[i + 1] << std::endl
                << "                rMin[" << i + 1 << "] = " << rInner[i + 1]
                << " -- rMax[" << i << "] = " << rOuter[i];
            //UUtils::Exception("UPolycone::UPolycone()", "GeomSolids0002",
                          //FatalErrorInArguments, 1, message.str().c_str());
#endif
            }
    }

    double rMin = rInner[i];

    double rMax = rOuter[i];
    double z = zPlane[i];

    // i has to be at least one to complete a section
    if (i > 0)
    {
     if (((z > prevZ)&&(dirZ>0))||((z < prevZ)&&(dirZ<0)))
      {
        if (dirZ*(z-prevZ)< 0)
        {
     
          //std::ostringstream message;
#ifndef VECGEOM_NVCC
            std::cerr << "Cannot create a Polycone with different Z directions.Use GenericPolycone."
                  << std::endl
                  << "              ZPlane is changing direction  !" << std::endl
                  << "  zPlane[0] = " << zPlane[0]
                  << " -- zPlane[1] = " << zPlane[1] << std::endl
                  << "  zPlane[" << i - 1 << "] = " << zPlane[i - 1]
                  << " -- rPlane[" << i << "] = " << zPlane[i];
          //UUtils::Exception("UPolycone::UPolycone()", "GeomSolids0002",
                            //FatalErrorInArguments, 1, message.str().c_str());
#endif
        }


        // here determine section shape
        // was: VUSolid* solid;
        // now
        UnplacedCone * solid;

        double dz = (z - prevZ) / 2;

        //bool tubular = (rMin == prevRmin && prevRmax == rMax);

//        if (fNumSides == 0)
        //{
//          if (tubular)
          //{
            solid = new UnplacedCone(prevRmin, prevRmax, rMin, rMax, dz, phiStart, phiTotal);
          //}
          //else
          //{
//            solid = new UCons("", prevRmin, prevRmax, rMin, rMax, dz, phiStart, phiTotal);
          //}
        //}

        fZs.push_back(z);
        int zi = fZs.size() - 1;
        double shift = fZs[zi - 1] + 0.5 * (fZs[zi] - fZs[zi - 1]);

        PolyconeSection section;
        section.shift = shift;
//        section.tubular = tubular;
        section.solid = solid;
        if( false /*tubular*/)
        {
          if (rMax < RMaxextent) { section.convex = false;}
          else { section.convex = true;}
        }
        else
        {
          if ((rMax<prevRmax)||(rMax < RMaxextent)||(prevRmax < RMaxextent))
            { section.convex = false;}
          else
            { section.convex = true;}
        }
        fSections.push_back(section);
      }
    }
    else{ // for i == 0 just push back first z plane
        fZs.push_back(z);
    }

    prevZ = z;
    prevRmin = rMin;
    prevRmax = rMax;
  } // end loop over Nz

  //
  // Build RZ polygon using special PCON/PGON GEANT3 constructor
  //

//
//  UReduciblePolygon* rz = new UReduciblePolygon(rInner, rOuter, zPlane, numZPlanes);
//
//  double mxy = rz->Amax();
////  double alfa = UUtils::kPi / fNumSides;
//
//  double r = rz->Amax();
////
//// Perform checks of rz values
////
//  if (rz->Amin() < 0.0)
//  {
//     std::ostringstream message;
//     message << "Illegal input parameters - " << GetName() << std::endl
//             << "        All R values must be >= 0 !";
//     UUtils::Exception("UPolycone::Init()", "GeomSolids0002",
//               FatalErrorInArguments,1, message.str().c_str());
//  }


   /*
  if (fNumSides != 0)
  {
    // mxy *= std::sqrt(2.0); // this is old and wrong, works only for n = 4
    // double k = std::tan(alfa) * mxy;
    double l = mxy / std::cos(alfa);
    mxy = l;
    r = l;
  }
  */

 // mxy += fgTolerance;

  //fBox.Set(mxy, mxy, (rz->Bmax() - rz->Bmin()) / 2);

  //
  // Make enclosingCylinder
  //

  //enclosingCylinder = new UEnclosingCylinder(r, rz->Bmax(), rz->Bmin(), phiIsOpen, phiStart, phiTotal);

  //delete rz;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
    VECGEOM_CUDA_HEADER_DEVICE
    VPlacedVolume* UnplacedPolycone::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
    #ifdef VECGEOM_NVCC
                                   const int id,
    #endif
                                   VPlacedVolume *const placement )
    {

    if (placement) {
        new(placement) SpecializedPolycone<transCodeT, rotCodeT>(logical_volume,
                                                                transformation
    #ifdef VECGEOM_NVCC
                                                              , NULL, id
    #endif
                                                              );
        return placement;
      }
    return new SpecializedPolycone<transCodeT, rotCodeT>(logical_volume,
                                                      transformation
    #ifdef VECGEOM_NVCC
                                , NULL, id
    #endif
                                );

    }

    void UnplacedPolycone::Print() const {
    printf("UnplacedPolycone {%.2f, %.2f, %d}\n",
            fStartPhi, fDeltaPhi, fNz);
    printf("have %d size Z\n", fZs.size());
    printf("------- z planes follow ---------\n");
    for(int p = 0; p < fZs.size(); ++p)
    {
        printf(" plane %d at z pos %lf\n", p, fZs[p]);
    }

    printf("have %d size fSections\n", fSections.size());
    printf("------ sections follow ----------\n");
    for(int s=0;s<GetNSections();++s)
    {
        printf("## section %d, shift %lf\n", s, fSections[s].shift);
        fSections[s].solid->Print();
        printf("\n");
    }
    }

    void UnplacedPolycone::Print(std::ostream &os) const {
    os << "UnplacedPolycone output to string not implemented\n";
    }


    VECGEOM_CUDA_HEADER_DEVICE
    VPlacedVolume* UnplacedPolycone::SpecializedVolume(
          LogicalVolume const *const volume,
          Transformation3D const *const transformation,
          const TranslationCode trans_code, const RotationCode rot_code,
    #ifdef VECGEOM_NVCC
          const int id,
    #endif
          VPlacedVolume *const placement ) const
    {

        // TODO: for the Polycone this might be overkill
        return VolumeFactory::CreateByTransformation<
            UnplacedPolycone>(volume, transformation, trans_code, rot_code,
      #ifdef VECGEOM_NVCC
                                    id,
      #endif
                                    placement);
    }

#ifdef VECGEOM_CUDA_INTERFACE

    DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu() const {
        return  CopyToGpuImpl<UnplacedPolycone>();
    }

    DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const {

        // idea: reconstruct defining arrays: copy them to GPU; then construct the UnplacedPolycon object from scratch
        // on the GPU
        std::vector<Precision> rmin, z, rmax;
        ReconstructSectionArrays(z,rmin,rmax);

    // somehow this does not work:
    //        Precision *z_gpu_ptr = AllocateOnGpu<Precision>( (z.size() + rmin.size() + rmax.size())*sizeof(Precision) );
    //        Precision *rmin_gpu_ptr = z_gpu_ptr + sizeof(Precision)*z.size();
    //        Precision *rmax_gpu_ptr = rmin_gpu_ptr + sizeof(Precision)*rmin.size();

    Precision *z_gpu_ptr = AllocateOnGpu<Precision>( z.size()*sizeof(Precision) );
    Precision *rmin_gpu_ptr = AllocateOnGpu<Precision>( rmin.size()*sizeof(Precision) );
    Precision *rmax_gpu_ptr = AllocateOnGpu<Precision>( rmax.size()*sizeof(Precision) );

        vecgeom::CopyToGpu(&z[0], z_gpu_ptr, sizeof(Precision)*z.size());
        vecgeom::CopyToGpu(&rmin[0], rmin_gpu_ptr, sizeof(Precision)*rmin.size());
        vecgeom::CopyToGpu(&rmax[0], rmax_gpu_ptr, sizeof(Precision)*rmax.size());

        DevicePtr<cuda::VUnplacedVolume> gpupolycon =  CopyToGpuImpl<UnplacedPolycone>(gpu_ptr,
                fStartPhi, fDeltaPhi, fNz, rmin_gpu_ptr, rmax_gpu_ptr, z_gpu_ptr);

        // remove temporary space from GPU
        FreeFromGpu(z_gpu_ptr);
        FreeFromGpu(rmin_gpu_ptr);
        FreeFromGpu(rmax_gpu_ptr);


        return gpupolycon;
    }

 #endif // VECGEOM_CUDA_INTERFACE

#ifndef VECGEOM_NVCC

     //#ifdef VECGEOM_USOLIDS
/////////////////////////////////////////////////////////////////////////
//
// GetPointOnSurface
//
// GetPointOnCone
//
// Auxiliary method for Get Point On Surface
//

Vector3D<Precision> UnplacedPolycone::GetPointOnCone(Precision fRmin1, Precision fRmax1,
                                   Precision fRmin2, Precision fRmax2,
                                   Precision zOne,   Precision zTwo,
                                   Precision& totArea) const
{
  // declare working variables
  //
  Precision Aone, Atwo, Afive, phi, zRand, fDPhi, cosu, sinu;
  Precision rRand1, rmin, rmax, chose, rone, rtwo, qone, qtwo;
  Precision fDz = (zTwo - zOne) / 2., afDz = std::fabs(fDz);
  Vector3D<Precision> point, offset = Vector3D<Precision>(0., 0., 0.5 * (zTwo + zOne));
  fDPhi = GetDeltaPhi();
  rone = (fRmax1 - fRmax2) / (2.*fDz);
  rtwo = (fRmin1 - fRmin2) / (2.*fDz);
  if (fRmax1 == fRmax2)
  {
    qone = 0.;
  }
  else
  {
    qone = fDz * (fRmax1 + fRmax2) / (fRmax1 - fRmax2);
  }
  if (fRmin1 == fRmin2)
  {
    qtwo = 0.;
  }
  else
  {
    qtwo = fDz * (fRmin1 + fRmin2) / (fRmin1 - fRmin2);
  }
  Aone   = 0.5 * fDPhi * (fRmax2 + fRmax1) * ((fRmin1 - fRmin2)*(fRmin1 - fRmin2) + (zTwo - zOne)*(zTwo - zOne));
  Atwo   = 0.5 * fDPhi * (fRmin2 + fRmin1) * ((fRmax1 - fRmax2)*(fRmax1 - fRmax2) + (zTwo - zOne)*(zTwo - zOne));
  Afive  = fDz * (fRmax1 - fRmin1 + fRmax2 - fRmin2);
  totArea = Aone + Atwo + 2.*Afive;

     phi  = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosu = std::cos(phi);
  sinu = std::sin(phi);


  if (GetDeltaPhi() >= kTwoPi )
  {
    Afive = 0;
  }
  chose = RNG::Instance().uniform(0., Aone + Atwo + 2.*Afive);
  if ((chose >= 0) && (chose < Aone))
  {
    if (fRmax1 != fRmax2)
    {
      zRand = RNG::Instance().uniform(-1.*afDz, afDz);
      point = Vector3D<Precision>(rone * cosu * (qone - zRand),
                       rone * sinu * (qone - zRand), zRand);
    }
    else
    {
      point = Vector3D<Precision>(fRmax1 * cosu, fRmax1 * sinu,
                       RNG::Instance().uniform(-1.*afDz, afDz));

    }
  }
  else if (chose >= Aone && chose < Aone + Atwo)
  {
    if (fRmin1 != fRmin2)
    {
      zRand = RNG::Instance().uniform(-1.*afDz, afDz);
      point = Vector3D<Precision>(rtwo * cosu * (qtwo - zRand),
                       rtwo * sinu * (qtwo - zRand), zRand);

    }
    else
    {
      point = Vector3D<Precision>(fRmin1 * cosu, fRmin1 * sinu,
                       RNG::Instance().uniform(-1.*afDz, afDz));
    }
  }
  else if ((chose >= Aone + Atwo + Afive) && (chose < Aone + Atwo + 2.*Afive))
  {
    zRand  = RNG::Instance().uniform(-afDz, afDz);
    rmin   = fRmin2 - ((zRand - fDz) / (2.*fDz)) * (fRmin1 - fRmin2);
    rmax   = fRmax2 - ((zRand - fDz) / (2.*fDz)) * (fRmax1 - fRmax2);
    rRand1 = std::sqrt(RNG::Instance().uniform(0.,1.)  * (rmax*rmax - rmin*rmin) +rmin*rmin);
     point  = Vector3D<Precision>(rRand1 * std::cos(GetStartPhi()),
    rRand1 * std::sin(GetStartPhi()), zRand);
  }
  else
  {
    zRand  = RNG::Instance().uniform(-1.*afDz, afDz);
    rmin   = fRmin2 - ((zRand - fDz) / (2.*fDz)) * (fRmin1 - fRmin2);
    rmax   = fRmax2 - ((zRand - fDz) / (2.*fDz)) * (fRmax1 - fRmax2);
    rRand1 = std::sqrt(RNG::Instance().uniform(0.,1.)  * (rmax*rmax - rmin*rmin) + rmin*rmin);
     point  = Vector3D<Precision>(rRand1 * std::cos(GetEndPhi()),
    rRand1 * std::sin(GetEndPhi()), zRand);

  }

  return point + offset;
}


//
// GetPointOnTubs
//
// Auxiliary method for GetPoint On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnTubs(Precision fRMin, Precision fRMax,
                                   Precision zOne,  Precision zTwo,
                                   Precision& totArea) const
{
  Precision xRand, yRand, zRand, phi, cosphi, sinphi, chose,
         aOne, aTwo, aFou, rRand, fDz, fSPhi, fDPhi;
  fDz = std::fabs(0.5 * (zTwo - zOne));
  fSPhi = GetStartPhi();
  fDPhi = GetDeltaPhi();

  aOne = 2.*fDz * fDPhi * fRMax;
  aTwo = 2.*fDz * fDPhi * fRMin;
  aFou = 2.*fDz * (fRMax - fRMin);
  totArea = aOne + aTwo + 2.*aFou;
     phi    = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  rRand  = fRMin + (fRMax - fRMin) * std::sqrt(RNG::Instance().uniform(0.,1.) );

     if (GetDeltaPhi() >= 2 * kPi)
    aFou = 0;

  chose  = RNG::Instance().uniform(0., aOne + aTwo + 2.*aFou);
  if ((chose >= 0) && (chose < aOne))
  {
    xRand = fRMax * cosphi;
    yRand = fRMax * sinphi;
    zRand = RNG::Instance().uniform(-1.*fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  }
  else if ((chose >= aOne) && (chose < aOne + aTwo))
  {
    xRand = fRMin * cosphi;
    yRand = fRMin * sinphi;
    zRand = RNG::Instance().uniform(-1.*fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  }
  else if ((chose >= aOne + aTwo) && (chose < aOne + aTwo + aFou))
  {
    xRand = rRand * std::cos(fSPhi + fDPhi);
    yRand = rRand * std::sin(fSPhi + fDPhi);
    zRand = RNG::Instance().uniform(-1.*fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  }

  // else

  xRand = rRand * std::cos(fSPhi + fDPhi);
  yRand = rRand * std::sin(fSPhi + fDPhi);
  zRand = RNG::Instance().uniform(-1.*fDz, fDz);
  return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
}


//
// GetPointOnRing
//
// Auxiliary method for GetPoint On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnRing(Precision fRMin1, Precision fRMax1,
                                   Precision fRMin2, Precision fRMax2,
                                   Precision zOne) const
{
  Precision xRand, yRand, phi, cosphi, sinphi, rRand1, rRand2, A1, Atot, rCh;
     phi    = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);

  if (fRMin1 == fRMin2)
  {
    rRand1 = fRMin1;
    A1 = 0.;
  }
  else
  {
    rRand1 = RNG::Instance().uniform(fRMin1, fRMin2);
    A1 = std::fabs(fRMin2 * fRMin2 - fRMin1 * fRMin1);
  }
  if (fRMax1 == fRMax2)
  {
    rRand2 = fRMax1;
    Atot = A1;
  }
  else
  {
    rRand2 = RNG::Instance().uniform(fRMax1, fRMax2);
    Atot   = A1 + std::fabs(fRMax2 * fRMax2 - fRMax1 * fRMax1);
  }
  rCh   = RNG::Instance().uniform(0., Atot);

  if (rCh > A1)
  {
    rRand1 = rRand2;
  }

  xRand = rRand1 * cosphi;
  yRand = rRand1 * sinphi;

  return Vector3D<Precision>(xRand, yRand, zOne);
}


//
// GetPointOnCut
//
// Auxiliary method for Get Point On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnCut(Precision fRMin1, Precision fRMax1,
                                  Precision fRMin2, Precision fRMax2,
                                  Precision zOne,  Precision zTwo,
                                  Precision& totArea) const
{
  if (zOne == zTwo)
  {
    return GetPointOnRing(fRMin1, fRMax1, fRMin2, fRMax2, zOne);
  }
  if ((fRMin1 == fRMin2) && (fRMax1 == fRMax2))
  {
    return GetPointOnTubs(fRMin1, fRMax1, zOne, zTwo, totArea);
  }
  return GetPointOnCone(fRMin1, fRMax1, fRMin2, fRMax2, zOne, zTwo, totArea);
}



//
// GetPointOnSurface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnSurface() const
{
  Precision Area = 0, totArea = 0, Achose1 = 0, Achose2 = 0, phi, cosphi, sinphi, rRand;
  int i = 0;
  int numPlanes = GetNSections();

  phi = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  std::vector<Precision> areas;
  PolyconeSection const & sec0 = GetSection(0);
  areas.push_back(kPi * (sec0.solid->GetRmax1()*sec0.solid->GetRmax1()
    - sec0.solid->GetRmin1()*sec0.solid->GetRmin1()));
    rRand = sec0.solid->GetRmin1() +
   ((sec0.solid->GetRmax1() - sec0.solid->GetRmin1())
           * std::sqrt(RNG::Instance().uniform(0.,1.) ));


  areas.push_back(kPi * (sec0.solid->GetRmax1()*sec0.solid->GetRmax1()
                                 - sec0.solid->GetRmin1()*sec0.solid->GetRmin1()));

  for (i = 0; i < numPlanes - 1; i++)
  {
     PolyconeSection const & sec = GetSection(i);
     Area = (sec.solid->GetRmin1() + sec.solid->GetRmin2())
          * std::sqrt((sec.solid->GetRmin1()-
          sec.solid->GetRmin2())*(sec.solid->GetRmin1()-
          sec.solid->GetRmin2())+
                 4.*sec.solid->GetDz()*sec.solid->GetDz());

     Area += (sec.solid->GetRmax1() + sec.solid->GetRmax2())
           * std::sqrt((sec.solid->GetRmax1()-
             sec.solid->GetRmax2())*(sec.solid->GetRmax1()-
             sec.solid->GetRmax2())+
                 4.*sec.solid->GetDz()*sec.solid->GetDz());

     Area *= 0.5 * GetDeltaPhi();

     if (GetDeltaPhi() < kTwoPi)
      {
      Area += std::fabs(2*sec.solid->GetDz()) *
        ( sec.solid->GetRmax1()
            + sec.solid->GetRmax2()
           - sec.solid->GetRmin1()
           - sec.solid->GetRmin2());
      }
     
    
    areas.push_back(Area);
    totArea += Area;
  }
  PolyconeSection const & secn = GetSection(numPlanes - 1);
  areas.push_back(kPi * kPi * (secn.solid->GetRmax2()*secn.solid->GetRmax2()
                  - secn.solid->GetRmin2()*secn.solid->GetRmin2()));

   
  totArea += (areas[0] + areas[numPlanes]);
  Precision chose = RNG::Instance().uniform(0., totArea);

  if ((chose >= 0.) && (chose < areas[0]))
  {
    return Vector3D<Precision>(rRand * cosphi, rRand * sinphi,
                    fZs[0]);
  }

  for (i = 0; i < numPlanes - 1; i++)
  {
    Achose1 += areas[i];
    Achose2 = (Achose1 + areas[i + 1]);
    if (chose >= Achose1 && chose < Achose2)
    {
      PolyconeSection const & sec = GetSection(i);
     return GetPointOnCut(sec.solid->GetRmin1(),
                              sec.solid->GetRmax1(),
                              sec.solid->GetRmin2(),
                              sec.solid->GetRmax2(),
                              fZs[i],
                              fZs[i + 1], Area);
    }
  }

     rRand = secn.solid->GetRmin2() +
         ((secn.solid->GetRmax2() - secn.solid->GetRmin2())
           * std::sqrt(RNG::Instance().uniform(0.,1.) ));

  return Vector3D<Precision>(rRand * cosphi, rRand * sinphi,
                             fZs[numPlanes]);
}
#endif


bool UnplacedPolycone::Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const {
     bool valid = true ;
     int index = GetSectionIndex(point.z());
    
     if(index < 0)
      {valid = true;
     if(index == -1) norm = Vector3D<Precision>(0.,0.,-1.);
     if(index == -2)  norm  = Vector3D<Precision>(0.,0.,1.);
         return valid;
      } 
     //PolyconeSection const & sec = GetSection(index);
      //TODO Normal to Section, need Normal from Cone impemenation
     valid = false;//sec.solid->Normal(point,norm);
     return valid;

}    

Precision UnplacedPolycone::SurfaceArea() const{
    Precision Area = 0, totArea = 0;
    int i = 0;
    int numPlanes = GetNSections();
    Precision fSurfaceArea = 0;
    
    Vector<Precision> areas;       // (numPlanes+1);
   
    PolyconeSection const & sec0 = GetSection(0);
     areas.push_back(kPi * (sec0.solid->GetRmax1()*sec0.solid->GetRmax1()
     - sec0.solid->GetRmin1()*sec0.solid->GetRmin1()));

    for (i = 0; i < numPlanes - 1; i++)
    {
     PolyconeSection const & sec = GetSection(i);
     Area = (sec.solid->GetRmin1() + sec.solid->GetRmin2())
             * std::sqrt((sec.solid->GetRmin1()-
         sec.solid->GetRmin2())*(sec.solid->GetRmin1()-
         sec.solid->GetRmin2())+
                 4.*sec.solid->GetDz()*sec.solid->GetDz());

      Area += (sec.solid->GetRmax1() + sec.solid->GetRmax2())
             * std::sqrt((sec.solid->GetRmax1()-
             sec.solid->GetRmax2())*(sec.solid->GetRmax1()-
             sec.solid->GetRmax2())+
                 4.*sec.solid->GetDz()*sec.solid->GetDz());

     Area *= 0.5 * GetDeltaPhi();

     if (GetDeltaPhi() < kTwoPi)
      {
     Area += std::fabs(2*sec.solid->GetDz()) *
       ( sec.solid->GetRmax1()
       + sec.solid->GetRmax2()
       - sec.solid->GetRmin1()
       - sec.solid->GetRmin2());
      }
      areas.push_back(Area);
      totArea += Area;
    }
     PolyconeSection const & secn = GetSection(numPlanes - 1);
     areas.push_back(kPi * kPi * (secn.solid->GetRmax2()*secn.solid->GetRmax2()
     - secn.solid->GetRmin2()*secn.solid->GetRmin2()));

     totArea += (areas[0] + areas[numPlanes]);
     fSurfaceArea = totArea;

  

  return fSurfaceArea;


}
 void UnplacedPolycone::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const {

    int i = 0;
    Precision maxR = 0;
    
    for (i = 0; i < GetNSections(); i++)
    {
     PolyconeSection const & sec = GetSection(i);
     if(maxR > sec.solid->GetRmax1())  maxR = sec.solid->GetRmax1(); 
     if(maxR > sec.solid->GetRmax2())  maxR = sec.solid->GetRmax2(); 
    }
    
     aMin.x() = -maxR;
         aMin.y() = -maxR;
         aMin.z() = fZs[0];
         aMax.x() = maxR;
         aMax.y() = maxR;
         aMax.z() = fZs[GetNSections()];
         
}
     //#endif

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedPolycone>::SizeOf();
template void DevicePtr<cuda::UnplacedPolycone>::Construct(
        Precision, Precision, int, Precision *, Precision *, Precision *) const;

} // End cxx namespace

#endif

} // end global namespace
