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
  double prevZ = 0, prevRmax = 0, prevRmin = 0;
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
      if (z > prevZ)
      {
        if (dirZ < 0)
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

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedPolycone>::SizeOf();
template void DevicePtr<cuda::UnplacedPolycone>::Construct(
        Precision, Precision, int, Precision *, Precision *, Precision *) const;

} // End cxx namespace

#endif

} // end global namespace
