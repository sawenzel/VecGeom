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
#include <iostream>
#include <cstdio>

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
  //Convertion for angles

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

    if (rInner[j]>rOuter[j])
    {

      std::cerr << "Cannot create Polycone with rInner > rOuter for the same Z"
              << "\n"
              << "        rInner > rOuter for the same Z !\n"
              << "        rMin[" << j << "] = " << rInner[j]
              << " -- rMax[" << j << "] = " << rOuter[j];
      // UUtils::Exception("UPolycone::UPolycone()", "GeomSolids0002",
        //                 FatalErrorInArguments, 1, message.str().c_str());
    }
  }

  //
  double prevZ = 0, prevRmax = 0, prevRmin = 0;
  int dirZ = 1;
  if (zPlane[1] < zPlane[0])dirZ = -1;
//  int curSolid = 0;

  int i;
  for (i = 0; i < numZPlanes; i++)
  {
    if ((i < numZPlanes - 1) && (zPlane[i] == zPlane[i + 1]))
    {
      if ((rInner[i]  > rOuter[i + 1])
          || (rInner[i + 1] > rOuter[i]))
      {
        std::cerr << "Cannot create a Polycone with no contiguous segments."
                << std::endl
                << "                Segments are not contiguous !" << std::endl
                << "                rMin[" << i << "] = " << rInner[i]
                << " -- rMax[" << i + 1 << "] = " << rOuter[i + 1] << std::endl
                << "                rMin[" << i + 1 << "] = " << rInner[i + 1]
                << " -- rMax[" << i << "] = " << rOuter[i];
        //UUtils::Exception("UPolycone::UPolycone()", "GeomSolids0002",
                          //FatalErrorInArguments, 1, message.str().c_str());
      }
    }



    double rMin = rInner[i];
    double rMax = rOuter[i];
    double z = zPlane[i];

    if (i > 0)
    {
      if (z > prevZ)
      {
        if (dirZ < 0)
        {
          //std::ostringstream message;
          std::cerr << "Cannot create a Polycone with different Z directions.Use GenericPolycone."
                  << std::endl
                  << "              ZPlane is changing direction  !" << std::endl
                  << "  zPlane[0] = " << zPlane[0]
                  << " -- zPlane[1] = " << zPlane[1] << std::endl
                  << "  zPlane[" << i - 1 << "] = " << zPlane[i - 1]
                  << " -- rPlane[" << i << "] = " << zPlane[i];
          //UUtils::Exception("UPolycone::UPolycone()", "GeomSolids0002",
                            //FatalErrorInArguments, 1, message.str().c_str());



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
      else
      {
        ;// i = i;
      }
    }
    else fZs.push_back(z);


    prevZ = z;
    prevRmin = rMin;
    prevRmax = rMax;
  }

  //fMaxSection = fZs.size() - 1;
  fMaxSection = fZs.size() - 2;

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
    printf("UnplacedPolycone {%.2f, %.2f, %.2d}",
            fStartPhi, fDeltaPhi, fNz);
    printf("------ sections follow ----------\n");
    for(int s=0;s<fMaxSection;++s)
    {
        fSections[s].solid->Print();
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


}} // end namespace
