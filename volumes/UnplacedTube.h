/// @file UnplacedTube.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDTUBE_H_
#define VECGEOM_VOLUMES_UNPLACEDTUBE_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"
#include "base/RNG.h"
#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/Wedge.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedTube; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedTube );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTube : public VUnplacedVolume, public AlignedBase {

private:
  // tube defining parameters
  Precision fRmin, fRmax, fZ, fSphi, fDphi;

  // cached values
  Precision fRmin2, fRmax2, fAlongPhi1x, fAlongPhi1y, fAlongPhi2x, fAlongPhi2y;
  Precision fTolIrmin2, fTolOrmin2, fTolIrmax2, fTolOrmax2, fTolIz, fTolOz;
  Precision fTolIrmin, fTolOrmin, fTolIrmax, fTolOrmax;
  Wedge fPhiWedge;

  VECGEOM_CUDA_HEADER_BOTH
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y) {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void calculateCached() {
    fTolIz = fZ - kTolerance;
    fTolOz = fZ + kTolerance;

    fRmin2 = fRmin * fRmin;
    fRmax2 = fRmax * fRmax;

 	fTolOrmin = (fRmin - kTolerance);
    fTolIrmin = (fRmin + kTolerance);
	fTolOrmin2 = fTolOrmin * fTolOrmin;
	fTolIrmin2 = fTolIrmin * fTolIrmin;

	fTolOrmax = (fRmax + kTolerance);
	fTolIrmax = (fRmax - kTolerance);
	fTolOrmax2 = fTolOrmax * fTolOrmax;
	fTolIrmax2 = fTolIrmax * fTolIrmax;

    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTube(const Precision rmin, const Precision rmax, const Precision z,
               const Precision sphi, const Precision dphi) : fRmin(rmin), fRmax(rmax),
     fZ(z), fSphi(sphi), fDphi(dphi),
fRmin2(0),
fRmax2(0),
fAlongPhi1x(0),
fAlongPhi1y(0),
fAlongPhi2x(0),
fAlongPhi2y(0),
fTolIrmin2(0),
fTolOrmin2(0),
fTolIrmax2(0),
fTolOrmax2(0),
fTolIz(0),
fTolOz(0),
fPhiWedge(dphi,sphi)
{
    calculateCached();
  }

  VECGEOM_CUDA_HEADER_BOTH
     UnplacedTube(UnplacedTube const &other) : fRmin(other.fRmin), fRmax(other.fRmax), fZ(other.fZ), fSphi(other.fSphi), fDphi(other.fDphi),
fRmin2(other.fRmin2),
fRmax2(other.fRmax2),
fAlongPhi1x(other.fAlongPhi1x),
fAlongPhi1y(other.fAlongPhi1y),
fAlongPhi2x(other.fAlongPhi2x),
fAlongPhi2y(other.fAlongPhi2y),
fTolIrmin2(other.fTolIrmin2),
fTolOrmin2(other.fTolOrmin2),
fTolIrmax2(other.fTolIrmax2),
fTolOrmax2(other.fTolOrmax2),
fTolIz(other.fTolIz),
fTolOz(other.fTolOz),
fPhiWedge(other.fDphi,other.fSphi)
{  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin() const { return fRmin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax() const { return fRmax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return fZ; }

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
  Precision tolIz() const { return fTolIz; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOz() const { return fTolOz; }

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
  Wedge const & GetWedge() const { return fPhiWedge; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision volume() const {
    return fZ * (fRmax2 - fRmin2) * fDphi;
  }

//#if !defined(VECGEOM_NVCC)
  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const;

#ifndef VECGEOM_NVCC
  Vector3D<Precision> GetPointOnSurface() const;

  //VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const {
      return volume();
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea () const {
      return GetTopArea() + GetLateralPhiArea() + GetLateralROutArea() + GetLateralRInArea();
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetTopArea() const {            // Abhijit:: this is top and bottom circular area of tube
      return 2*0.5*(fRmax2 - fRmin2) * fDphi;
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetLateralPhiArea() const {     // Abhijit:: this is vertical Phi_start and phi_end opening
      // factor of 2 since fZ is half length
      return (fDphi < vecgeom::kTwoPi) ? 4.*fZ*(fRmax - fRmin) : 0.;
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetLateralRInArea() const {    // Abhijit:: this is Inner surface of tube along Z
      // factor of 2 since fZ is half length
      return 2.*fZ*fRmin*fDphi;
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetLateralROutArea() const {  // Abhijit:: this is Outer surface of tube along Z
      // factor of 2 since fZ is half length
      return 2.*fZ*fRmax*fDphi;
  }

  //  This computes where the random point would be placed
  // 1::rTop, 2::rBot, 3::phiLeft, 4::phiRight, 5::zIn, 6::zOut
  //VECGEOM_CUDA_HEADER_BOTH
    int ChooseSurface() const;

	bool Normal(Vector3D<Precision>const& point, Vector3D<Precision>& normal) const;

#endif

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
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedTube>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
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

} } // end global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_UNPLACEDTUBE_H_

