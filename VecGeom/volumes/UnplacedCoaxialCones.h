/// @file UnplacedCoaxialCones.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDCOAXIALCONES_H_
#define VECGEOM_VOLUMES_UNPLACEDCOAXIALCONES_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/CoaxialConesStruct.h" // the pure CoaxialCones struct
#include "VecGeom/volumes/kernel/CoaxialConesImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedCoaxialCones;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedCoaxialCones);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedCoaxialCones : public UnplacedVolumeImplHelper<CoaxialConesImplementation>, public AlignedBase {

private:
  CoaxialConesStruct<Precision> fCoaxialCones;

public:
  /*
   * All the required Parametric CoaxialCones Constructor
   *
   */
  VECCORE_ATT_HOST_DEVICE
  UnplacedCoaxialCones() { ComputeBBox(); }

  VECCORE_ATT_HOST_DEVICE
  UnplacedCoaxialCones(unsigned int numOfCones, Precision *rmin1Vect, Precision *rmax1Vect, Precision *rmin2Vect,
                       Precision *rmax2Vect, Precision dz, Precision sphi, Precision dphi)
      : fCoaxialCones(numOfCones, rmin1Vect, rmax1Vect, rmin2Vect, rmax2Vect, dz, sphi, dphi)
  {
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::coaxialcones; }

  VECCORE_ATT_HOST_DEVICE
  CoaxialConesStruct<Precision> const &GetStruct() const { return fCoaxialCones; }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fCoaxialCones.fCubicVolume; }

  Precision SurfaceArea() const override { return fCoaxialCones.fSurfaceArea; }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    return fCoaxialCones.Normal(p, normal);
  }

  std::string GetEntityType() const { return "CoaxialCones"; }

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedCoaxialCones>::SizeOf(); }
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
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
