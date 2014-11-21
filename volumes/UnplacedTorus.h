/// @file UnplacedTorus.h

#ifndef VECGEOM_VOLUMES_UNPLACEDTORUS_H_
#define VECGEOM_VOLUMES_UNPLACEDTORUS_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedTube.h"
#include "volumes/Wedge.h"


namespace VECGEOM_NAMESPACE {

class UnplacedTorus : public VUnplacedVolume, public AlignedBase {

private:
  // torus defining parameters ( like G4torus )
    Precision fRmin; // outer radius of torus "tube"
    Precision fRmax; // innter radius of torus "tube"
    Precision fRtor; // bending radius of torus
    Precision fSphi; // start angle
    Precision fDphi; // delta angle of torus section
    Wedge     fPhiWedge; // the Phi bounding of the torus (not the cutout)

  // cached values
  Precision fRmin2, fRmax2, fRtor2, fAlongPhi1x, fAlongPhi1y, fAlongPhi2x, fAlongPhi2y;
  Precision fTolIrmin2, fTolOrmin2, fTolIrmax2, fTolOrmax2;
  // bounding tube
  UnplacedTube fBoundingTube;
 
  VECGEOM_CUDA_HEADER_BOTH
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y) {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void calculateCached() {
    fRmin2 = fRmin * fRmin;
    fRmax2 = fRmax * fRmax;
    fRtor2 = fRtor * fRtor;

    fTolOrmin2 = (fRmin - kTolerance)*(fRmin - kTolerance);
    fTolIrmin2 = (fRmin + kTolerance)*(fRmin + kTolerance);
    
    fTolOrmax2 = (fRmax + kTolerance)*(fRmax + kTolerance);
    fTolIrmax2 = (fRmax - kTolerance)*(fRmax - kTolerance);

    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

public:
  
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTorus(const Precision rmin, const Precision rmax, const Precision rtor,
               const Precision sphi, const Precision dphi) : fRmin(rmin), fRmax(rmax),
    fRtor(rtor), fSphi(sphi), fDphi(dphi), fPhiWedge(dphi,sphi), fBoundingTube(0, 1, 1, 0, dphi) {
    calculateCached(); 
    
    fBoundingTube = UnplacedTube(fRtor-fRmax - kTolerance,
    fRtor+fRmax + kTolerance, fRmax,
     sphi, dphi);
   
  }

//  VECGEOM_CUDA_HEADER_BOTH
//  UnplacedTorus(UnplacedTorus const &other) :
//  fRmin(other.fRmin), fRmax(other.fRmax), fRtor(other.fRtor), fSphi(other.fSphi), fDphi(other.fDphi),fBoundingTube(other.fBoundingTube) {
//    calculateCached();
//
//  }


    
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin() const { return fRmin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax() const { return fRmax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rtor() const { return fRtor; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision sphi() const { return fSphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dphi() const { return fDphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin2() const { return fRmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax2() const { return fRmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rtor2() const { return fRtor2; }

  VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    Wedge const & GetWedge() const { return fPhiWedge; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi1x() const { return fAlongPhi1x; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi1y() const { return fAlongPhi1y; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi2x() const { return fAlongPhi2x; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi2y() const { return fAlongPhi2y; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOrmin2() const { return fTolOrmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIrmin2() const { return fTolIrmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOrmax2() const { return fTolOrmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIrmax2() const { return fTolIrmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision volume() const {
    return fDphi*kPi*fRtor*(fRmax*fRmax-fRmin*fRmin);
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedTube const &GetBoundingTube() const { return fBoundingTube; }

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

private:

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

};

} // end global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTORUS_H_




