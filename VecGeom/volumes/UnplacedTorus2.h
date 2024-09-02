/// @file UnplacedTorus2.h

#ifndef VECGEOM_VOLUMES_UNPLACEDTORUS2_H_
#define VECGEOM_VOLUMES_UNPLACEDTORUS2_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/TorusStruct2.h"
#include "VecGeom/volumes/kernel/TorusImplementation2.h"
#include "VecGeom/volumes/Wedge.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTorus2;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTorus2);
// VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SIMDUnplacedTorus, typename);  // maybe needed w/TorusTypes

inline namespace VECGEOM_IMPL_NAMESPACE {

// Introduce Intermediate class ( so that we can do typecasting )
class UnplacedTorus2 : public UnplacedVolumeImplHelper<TorusImplementation2>, public AlignedBase {
private:
  // tube defining parameters
  TorusStruct2<Precision> fTorus;
  Wedge fPhiWedge; // the Phi bounding of the torus (not the cutout)

  // cached values
  Precision fRmin2, fRmax2, fRtor2, fAlongPhi1x, fAlongPhi1y, fAlongPhi2x, fAlongPhi2y;
  Precision fTolIrmin2, fTolOrmin2, fTolIrmax2, fTolOrmax2;
  // bounding tube
  GenericUnplacedTube fBoundingTube;

  VECCORE_ATT_HOST_DEVICE
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y)
  {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECCORE_ATT_HOST_DEVICE
  void calculateCached()
  {
    fRmin2 = fTorus.fRmin * fTorus.fRmin;
    fRmax2 = fTorus.fRmax * fTorus.fRmax;
    fRtor2 = fTorus.fRtor * fTorus.fRtor;

    fTolOrmin2 = (fTorus.fRmin - kTolerance) * (fTorus.fRmin - kTolerance);
    fTolIrmin2 = (fTorus.fRmin + kTolerance) * (fTorus.fRmin + kTolerance);

    fTolOrmax2 = (fTorus.fRmax + kTolerance) * (fTorus.fRmax + kTolerance);
    fTolIrmax2 = (fTorus.fRmax - kTolerance) * (fTorus.fRmax - kTolerance);

    GetAlongVectorToPhiSector(fTorus.fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fTorus.fSphi + fTorus.fDphi, fAlongPhi2x, fAlongPhi2y);
  }

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedTorus2(Precision const &_rmin, Precision const &_rmax, Precision const &_rtor, Precision const &_sphi,
                 Precision const &_dphi)
      : fTorus(_rmin, _rmax, _rtor, _sphi, _dphi), fPhiWedge(_dphi, _sphi), fBoundingTube(0, 1, 1, 0, _dphi)
  {
    calculateCached();

    fBoundingTube = GenericUnplacedTube(fTorus.fRtor - fTorus.fRmax - kTolerance,
                                        fTorus.fRtor + fTorus.fRmax + kTolerance, fTorus.fRmax, _sphi, _dphi);

    DetectConvexity();
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::torus; }

  VECCORE_ATT_HOST_DEVICE
  TorusStruct2<Precision> const &GetStruct() const { return fTorus; }

  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmin() const { return fTorus.fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmax() const { return fTorus.fRmax; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rtor() const { return fTorus.fRtor; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision sphi() const { return fTorus.fSphi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dphi() const { return fTorus.fDphi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmin2() const { return fRmin2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmax2() const { return fRmax2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rtor2() const { return fRtor2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Wedge const &GetWedge() const { return fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision alongPhi1x() const { return fAlongPhi1x; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision alongPhi1y() const { return fAlongPhi1y; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision alongPhi2x() const { return fAlongPhi2x; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision alongPhi2y() const { return fAlongPhi2y; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tolOrmin2() const { return fTolOrmin2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tolIrmin2() const { return fTolIrmin2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tolOrmax2() const { return fTolOrmax2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tolIrmax2() const { return fTolIrmax2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision volume() const
  {
    return fTorus.fDphi * kPi * fTorus.fRtor * (fTorus.fRmax * fTorus.fRmax - fTorus.fRmin * fTorus.fRmin);
  }

  VECCORE_ATT_HOST_DEVICE
  void SetRMin(Precision arg)
  {
    fTorus.fRmin = arg;
    calculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetRMax(Precision arg)
  {
    fTorus.fRmax = arg;
    calculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetRTor(Precision arg)
  {
    fTorus.fRtor = arg;
    calculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetSPhi(Precision arg)
  {
    fTorus.fSphi = arg;
    calculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetDPhi(Precision arg)
  {
    fTorus.fDphi = arg;
    calculateCached();
  }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override
  {
    Precision surfaceArea = fTorus.fDphi * kTwoPi * fTorus.fRtor * (fTorus.fRmax + fTorus.fRmin);
    if (fTorus.fDphi < kTwoPi) {
      surfaceArea = surfaceArea + kTwoPi * (fTorus.fRmax * fTorus.fRmax - fTorus.fRmin * fTorus.fRmin);
    }
    return surfaceArea;
  }

  Precision Capacity() const override { return volume(); }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const override;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  GenericUnplacedTube const &GetBoundingTube() const { return fBoundingTube; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Extent(Vector3D<Precision> &min, Vector3D<Precision> &max) const override { GetBoundingTube().Extent(min, max); }

  Vector3D<Precision> SamplePointOnSurface() const override;

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

  std::string GetEntityType() const { return "Torus"; }

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedTorus2>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

private:
#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           VPlacedVolume *const placement = NULL) const override;

#else
  __device__ virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation, const int id,
                                                      const int copy_no, const int child_id,
                                                      VPlacedVolume *const placement = NULL) const override;

#endif
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDTORUS2_H_
