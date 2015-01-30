/*
 * UnplacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */



#ifndef VECGEOM_VOLUMES_UNPLACEDCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDCONE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/Wedge.h"
#include <cmath>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedCone; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedCone );

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * Class representing an unplaced cone; Encapsulated parameters of a cone and
 * functions that do not depend on how the cone is placed in a reference frame
 *
 * The unplaced cone is represented by the following parameters
 *
 * Member Data:
 *
 *  fDz half length in z direction;  ( the cone has height 2*fDz )
 *  fRmin1  inside radius at  -fDz ( in internal coordinate system )
 *  fRmin2  inside radius at  +fDz
 *  fRmax1  outside radius at -fDz
 *  fRmax2  outside radius at +fDz
 *  fSPhi starting angle of the segment in radians
 *  fDPhi delta angle of the segment in radiansdz
 */
class UnplacedCone : public VUnplacedVolume, public AlignedBase {

private:
    VECGEOM_CUDA_HEADER_BOTH
      static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y) {
        x = std::cos(phi);
        y = std::sin(phi);
      }

  Precision fRmin1;
  Precision fRmax1;
  Precision fRmin2;
  Precision fRmax2;
  Precision fDz;
  Precision fSPhi;
  Precision fDPhi;
  Wedge     fPhiWedge; // the Phi bounding of the cone (not the cutout) -- will be able to get rid of the next angles

  // vectors characterizing the normals of phi planes
  // makes task to detect phi sektors very efficient
  Vector3D<Precision> fNormalPhi1;
  Vector3D<Precision> fNormalPhi2;
  Precision fAlongPhi1x;
  Precision fAlongPhi1y;
  Precision fAlongPhi2x;
  Precision fAlongPhi2y;

  // Some precomputed values to avoid divisions etc
  Precision fInnerSlope; // "gradient" of inner surface in z direction
  Precision fOuterSlope; // "gradient" of outer surface in z direction
  Precision fInnerOffset;
  Precision fOuterOffset;
  Precision fOuterSlopeSquare;
  Precision fInnerSlopeSquare;
  Precision fOuterOffsetSquare;
  Precision fInnerOffsetSquare;

public:

  /** momentarily include this fields because USolids algorithm needs those
  *
  *
  */
  Precision fSecRMin;
  Precision fSecRMax;
  Precision fInvSecRMin;
  Precision fInvSecRMax;


  Precision fCosCPhi;
  Precision fSinCPhi;
  Precision fCosSPhi;
  Precision fSinSPhi;
  Precision fCosEPhi;

  Precision fCosHDPhi;
Precision fSinEPhi;
  Precision fCosHDPhiIT;
  Precision fCosHDPhiOT;
  Precision fTanRMin;
  Precision fTanRMax;


  VECGEOM_CUDA_HEADER_BOTH
  // should be implemented in source file
  UnplacedCone(Precision rmin1,
               Precision rmax1,
               Precision rmin2,
               Precision rmax2,
               Precision dz, Precision phimin, Precision deltaphi) :
        fRmin1(rmin1),
        fRmax1(rmax1),
        fRmin2(rmin2),
        fRmax2(rmax2),
        fDz(dz),
        fSPhi(phimin),
        fDPhi(deltaphi),
        fPhiWedge(deltaphi,phimin),
        fNormalPhi1(),
        fNormalPhi2(),
        fAlongPhi1x(0),
        fAlongPhi1y(0),
        fAlongPhi2x(0),
        fAlongPhi2y(0),
        fInnerSlope(), // "gradient" of inner surface in z direction
        fOuterSlope(), // "gradient" of outer surface in z direction
        fInnerOffset(),
        fOuterOffset(),
        fOuterSlopeSquare(),
        fInnerSlopeSquare(),
        fOuterOffsetSquare(),
        fInnerOffsetSquare(),
	   fSecRMin(0),
	   fSecRMax(0),
	   fInvSecRMin(0),
	   fInvSecRMax(0),
   fCosCPhi(0),
   fSinCPhi(0),
   fCosSPhi(0),
   fSinSPhi(0),
   fCosEPhi(0),

   fCosHDPhi(0),
 fSinEPhi(0),
   fCosHDPhiIT(0),
   fCosHDPhiOT(0),
   fTanRMin(0),
   fTanRMax(0)
	      {

      // initialize trigonometry for USOLIDS impl
        double hDPhi = 0.5 * fDPhi;                    // half delta phi
        double cPhi = fSPhi + hDPhi;
        double ePhi = fSPhi + fDPhi;

        fSinCPhi   = std::sin(cPhi);
        fCosCPhi   = std::cos(cPhi);
        fCosHDPhi  = std::cos(hDPhi);
        fCosHDPhiIT = std::cos(hDPhi - 0.5 * kAngTolerance); // inner/outer tol half dphi
        fCosHDPhiOT = std::cos(hDPhi + 0.5 * kAngTolerance);
        fSinSPhi = std::sin(fSPhi);
        fCosSPhi = std::cos(fSPhi);
        fSinEPhi = std::sin(ePhi);
        fCosEPhi = std::cos(ePhi);
        fTanRMin = (fRmin2 - fRmin1) * 0.5 / fDz;

        fSecRMin = std::sqrt(1.0 + fTanRMin * fTanRMin);
        fInvSecRMin = 1./fSecRMin;

        fTanRMax = (fRmax2 - fRmax1) * 0.5 / fDz;

        fSecRMax = std::sqrt(1.0 + fTanRMax * fTanRMax);
        fInvSecRMax = 1./fSecRMax;

        // check this very carefully
       fInnerSlope = -(fRmin1 - fRmin2)/(2.*fDz);
       fOuterSlope = -(fRmax1 - fRmax2)/(2.*fDz);
       fInnerOffset = fRmin2 - fInnerSlope*fDz;
       fOuterOffset = fRmax2 - fOuterSlope*fDz;
       fOuterSlopeSquare = fOuterSlope*fOuterSlope;
       fInnerSlopeSquare = fInnerSlope*fInnerSlope;
       fOuterOffsetSquare = fOuterOffset*fOuterOffset;
       fInnerOffsetSquare = fInnerOffset*fInnerOffset;

       GetAlongVectorToPhiSector(fSPhi, fAlongPhi1x, fAlongPhi1y);
       GetAlongVectorToPhiSector(fSPhi + fDPhi, fAlongPhi2x, fAlongPhi2y);
       // calculate caches
       // the possible caches are one major difference between tube and cone

       // calculate caches
//       cacheRminSqr=dRmin1*dRmin1;
//       cacheRmaxSqr=dRmax1*dRmax1;

//       if ( dRmin1 > Utils::GetRadHalfTolerance() )
//           {
//              // CHECK IF THIS CORRECT ( this seems to be inversed with tolerance for ORmax
//              cacheTolORminSqr = (dRmin1 - Utils::GetRadHalfTolerance()) * (dRmin1 - Utils::GetRadHalfTolerance());
//              cacheTolIRminSqr = (dRmin1 + Utils::GetRadHalfTolerance()) * (dRmin1 + Utils::GetRadHalfTolerance());
//           }
//           else
//           {
//              cacheTolORminSqr = 0.0;
//              cacheTolIRminSqr = 0.0;
//           }
//
//           cacheTolORmaxSqr = (dRmax1 + Utils::GetRadHalfTolerance()) * (dRmax1 + Utils::GetRadHalfTolerance());
//           cacheTolIRmaxSqr = (dRmax1 - Utils::GetRadHalfTolerance()) * (dRmax1 - Utils::GetRadHalfTolerance());
//
//           // calculate normals
//           GeneralPhiUtils::GetNormalVectorToPhiPlane(dSPhi, normalPhi1, true);
//           GeneralPhiUtils::GetNormalVectorToPhiPlane(dSPhi + dDPhi, normalPhi2, false);
//
//           // get alongs
//           GeneralPhiUtils::GetAlongVectorToPhiPlane(dSPhi, alongPhi1);
//           GeneralPhiUtils::GetAlongVectorToPhiPlane(dSPhi + dDPhi, alongPhi2);
//
//           // normalPhi1.print();
//           // normalPhi2.print();
//        };
    }

    // public interfaces
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin1() const {return fRmin1;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax1() const {return fRmax1;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin2() const {return fRmin2;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax2() const {return fRmax2;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz() const {return fDz;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetSPhi() const {return fSPhi;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDPhi() const {return fDPhi;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision dphi() const {return fDPhi;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetInnerSlope() const {return fInnerSlope;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetOuterSlope() const {return fOuterSlope;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetInnerOffset() const {return fInnerOffset;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetOuterOffset() const {return fOuterOffset;}
    // these values could be cached
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetInnerSlopeSquare() const {return fInnerSlope*fInnerSlope;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetOuterSlopeSquare() const {return fOuterSlope*fOuterSlope;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetInnerOffsetSquare() const {return fInnerOffset*fInnerOffset;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetOuterOffsetSquare() const {return fOuterOffset*fOuterOffset;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision alongPhi1x() const { return fAlongPhi1x; }
    VECGEOM_CUDA_HEADER_BOTH
    Precision alongPhi1y() const { return fAlongPhi1y; }
    VECGEOM_CUDA_HEADER_BOTH
    Precision alongPhi2x() const { return fAlongPhi2x; }
    VECGEOM_CUDA_HEADER_BOTH
    Precision alongPhi2y() const { return fAlongPhi2y; }

    VECGEOM_CUDA_HEADER_BOTH
    Wedge const & GetWedge() const { return fPhiWedge; }

    VECGEOM_CUDA_HEADER_BOTH
    bool IsFullPhi() const { return fDPhi == kTwoPi; }

    VECGEOM_CUDA_HEADER_BOTH
    Precision Capacity() const {
        return (fDz * fDPhi / 3.)*(fRmax1*fRmax1+fRmax2*fRmax2+fRmax1*fRmax2-
                fRmin1*fRmin1-fRmin2*fRmin2-fRmin1*fRmin2);
    }

    virtual int memory_size() const { return sizeof(*this); }

    VECGEOM_CUDA_HEADER_BOTH
    virtual void Print() const;
    virtual void Print(std::ostream &os) const;

    VECGEOM_CUDA_HEADER_DEVICE
     virtual VPlacedVolume* SpecializedVolume(
         LogicalVolume const *const volume,
         Transformation3D const *const transformation,
         const TranslationCode trans_code, const RotationCode rot_code,
   #ifdef VECGEOM_NVCC
         const int id,
   #endif
         VPlacedVolume *const placement = NULL) const;

    template <TranslationCode transCodeT, RotationCode rotCodeT>
     VECGEOM_CUDA_HEADER_DEVICE
     static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                                  Transformation3D const *const transformation,
   #ifdef VECGEOM_NVCC
                                  const int id,
   #endif
                                  VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedCone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#ifdef VECGEOM_USOLIDS
  void Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
  {
      double max = fRmax1 > fRmax2 ? fRmax1 : fRmax2;
      aMin = Vector3D<Precision>(-max, -max, -fDz);
      aMax = Vector3D<Precision>(max, max, fDz);
  }

  Vector3D<Precision> GetPointOnSurface() const;

  Precision SurfaceArea()  const {
      double mmin, mmax, dmin, dmax;
      mmin = (fRmin1 + fRmin2) * 0.5;
      mmax = (fRmax1 + fRmax2) * 0.5;
      dmin = (fRmin2 - fRmin1);
      dmax = (fRmax2 - fRmax1);

      return fDPhi * (mmin * std::sqrt(dmin * dmin + 4 * fDz * fDz)
                                 + mmax * std::sqrt(dmax * dmax + 4 * fDz * fDz)
                                 + 0.5 * (fRmax1 * fRmax1 - fRmin1 * fRmin1
                                          + fRmax2 * fRmax2 - fRmin2 * fRmin2));
  }
#endif



};


} }  // End global namespace

#endif
