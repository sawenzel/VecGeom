/// \file UnplacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
#define VECGEOM_VOLUMES_UNPLACEDSPHERE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/SphereStruct.h"
#include "VecGeom/volumes/kernel/SphereImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedSphere;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedSphere);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedSphere : public UnplacedVolumeImplHelper<SphereImplementation>, public AlignedBase {

private:
  SphereStruct<Precision> fSphere;

public:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::sphere; }

  VECCORE_ATT_HOST_DEVICE
  SphereStruct<Precision> const &GetStruct() const { return fSphere; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  evolution::Wedge const &GetWedge() const { return fSphere.fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ThetaCone const &GetThetaCone() const { return fSphere.fThetaCone; }

  VECCORE_ATT_HOST_DEVICE
  UnplacedSphere(Precision pRmin, Precision pRmax, Precision pSPhi = 0., Precision pDPhi = kTwoPi,
                 Precision pSTheta = 0., Precision pDTheta = kPi);

  // specialized constructor for orb like instantiation
  VECCORE_ATT_HOST_DEVICE
  UnplacedSphere(Precision pR);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInsideRadius() const { return fSphere.fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadius() const { return fSphere.fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadius() const { return fSphere.fRmax; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStartPhiAngle() const { return fSphere.fSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDeltaPhiAngle() const { return fSphere.fDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStartThetaAngle() const { return fSphere.fSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDeltaThetaAngle() const { return fSphere.fDTheta; }

  // Functions to get Tolerance
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFRminTolerance() const { return fSphere.fRminTolerance; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetMKTolerance() const { return fSphere.mkTolerance; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetAngTolerance() const { return fSphere.kAngTolerance; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullSphere() const { return fSphere.fFullSphere; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullPhiSphere() const { return fSphere.fFullPhiSphere; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullThetaSphere() const { return fSphere.fFullThetaSphere; }

  // All angle related functions
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetHDPhi() const { return fSphere.hDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCPhi() const { return fSphere.cPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEPhi() const { return fSphere.ePhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinCPhi() const { return fSphere.sinCPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosCPhi() const { return fSphere.cosCPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinSPhi() const { return fSphere.sinSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosSPhi() const { return fSphere.cosSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinEPhi() const { return fSphere.sinEPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosEPhi() const { return fSphere.cosEPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetETheta() const { return fSphere.eTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinSTheta() const { return fSphere.sinSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosSTheta() const { return fSphere.cosSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanSTheta() const { return fSphere.tanSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanETheta() const { return fSphere.tanETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFabsTanSTheta() const { return fSphere.fabsTanSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFabsTanETheta() const { return fSphere.fabsTanETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanSTheta2() const { return fSphere.tanSTheta2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanETheta2() const { return fSphere.tanETheta2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinETheta() const { return fSphere.sinETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosETheta() const { return fSphere.cosETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosHDPhiOT() const { return fSphere.cosHDPhiOT; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosHDPhiIT() const { return fSphere.cosHDPhiIT; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetInsideRadius(Precision newRmin) { fSphere.SetInsideRadius(newRmin); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetInnerRadius(Precision newRmin) { SetInsideRadius(newRmin); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetOuterRadius(Precision newRmax) { fSphere.SetOuterRadius(newRmax); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetStartPhiAngle(Precision newSPhi, bool compute = true) { fSphere.SetStartPhiAngle(newSPhi, compute); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDeltaPhiAngle(Precision newDPhi) { fSphere.SetDeltaPhiAngle(newDPhi); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetStartThetaAngle(Precision newSTheta) { fSphere.SetStartThetaAngle(newSTheta); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDeltaThetaAngle(Precision newDTheta) { fSphere.SetDeltaThetaAngle(newDTheta); }

  // Old access functions
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return fSphere.fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return fSphere.fRmax; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return fSphere.fSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return fSphere.fDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSTheta() const { return fSphere.fSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDTheta() const { return fSphere.fDTheta; }

  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity();

  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea();

  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();

  Precision Capacity() const override { return fSphere.fCubicVolume; }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fSphere.fSurfaceArea; }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = SphereImplementation::Normal<Precision>(fSphere, point, valid);
    return valid;
  }

#ifndef VECCORE_CUDA

  Vector3D<Precision> SamplePointOnSurface() const override;

  std::string GetEntityType() const;

#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label = "") const;
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label = "") const;
#endif
#endif // VECCORE_CUDA

  void GetParametersList(int aNumber, Precision *aArray) const;

  std::ostream &StreamInfo(std::ostream &os) const;

  // VECCORE_ATT_HOST_DEVICE
  // Precision sqr(Precision x) {return x*x;};

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  // VECCORE_ATT_HOST_DEVICE
  virtual void Print(std::ostream &os) const final;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifndef VECCORE_CUDA

  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                Transformation3D const *const transformation,
                                                VPlacedVolume *const placement = NULL);

#else

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id,
                               VPlacedVolume *const placement = NULL);

  VECCORE_ATT_DEVICE static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                                   Transformation3D const *const transformation,
                                                                   const int id, const int copy_no, const int child_id,
                                                                   VPlacedVolume *const placement = NULL);

#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedSphere>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

private:
#ifndef VECCORE_CUDA

  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return CreateSpecializedVolume(volume, transformation, placement);
  }

#else

  VECCORE_ATT_DEVICE virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                              Transformation3D const *const transformation,
                                                              const int id, const int copy_no, const int child_id,
                                                              VPlacedVolume *const placement = NULL) const override
  {
    return CreateSpecializedVolume(volume, transformation, id, copy_no, child_id, placement);
  }

#endif
};

template <>
struct Maker<UnplacedSphere> {
  template <typename... ArgTypes>
  static UnplacedSphere *MakeInstance(Precision pRmin, Precision pRmax, Precision pSPhi, Precision pDPhi,
                                      Precision pSTheta, Precision pDTheta);
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
