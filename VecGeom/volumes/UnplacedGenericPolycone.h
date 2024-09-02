/// @file UnplacedGenericPolycone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDGENERICPOLYCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDGENERICPOLYCONE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/GenericPolyconeStruct.h"
#include "VecGeom/volumes/kernel/GenericPolyconeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"
#include "VecGeom/volumes/ReducedPolycone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedGenericPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedGenericPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedGenericPolycone : public UnplacedVolumeImplHelper<GenericPolyconeImplementation>, public AlignedBase {

private:
  GenericPolyconeStruct<Precision> fGenericPolycone;

  // Original Polycone Parameters
  Precision fSPhi;
  Precision fDPhi;
  int fNumRZ;
  Vector<Precision> fR;
  Vector<Precision> fZ;

  // Used for Extent
  Vector3D<Precision> fAMin;
  Vector3D<Precision> fAMax;

public:
  /*
   * All the required GenericPolycone Constructor
   *
   */
  VECCORE_ATT_HOST_DEVICE
  UnplacedGenericPolycone();

  VECCORE_ATT_HOST_DEVICE
  UnplacedGenericPolycone(Precision phiStart,  // initial phi starting angle
                          Precision phiTotal,  // total phi angle
                          int numRZ,           // number corners in r,z space (must be an even number)
                          Precision const *r,  // r coordinate of these corners
                          Precision const *z); // z coordinate of these corners

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::genericpolycone; }

  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeStruct<Precision> const &GetStruct() const { return fGenericPolycone; }

  /*
   * Required Getters and Setters
   *
   */

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return fSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return fDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetNumRz() const { return fNumRZ; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetR() const { return fR; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetZ() const { return fZ; }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const override;

  Precision Capacity() const override { return fGenericPolycone.fCubicVolume; }

  // Using the Generic implementation from UnplacedVolume
  Precision SurfaceArea() const override { return EstimateSurfaceArea(); }

  Vector3D<Precision> SamplePointOnSurface() const override;

  std::string GetEntityType() const { return "GenericPolycone"; }

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedGenericPolycone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  // this is the function called from the VolumeFactory
  // this may be specific to the shape
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement) const override;
#else
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id,
                               VPlacedVolume *const placement = NULL);
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation, const int id,
                                                      const int copy_no, const int child_id,
                                                      VPlacedVolume *const placement) const override;

#endif
};

using GenericUnplacedGenericPolycone = UnplacedGenericPolycone;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
