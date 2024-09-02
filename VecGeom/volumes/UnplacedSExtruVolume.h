#ifndef VECGEOM_VOLUMES_UNPLACED_SEXTRUVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACED_SEXTRUVOLUME_H_

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/PolygonalShell.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/SExtruImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedSExtruVolume;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedSExtruVolume);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedSExtruVolume : public UnplacedVolumeImplHelper<SExtruImplementation>, public AlignedBase {

private:
  PolygonalShell fPolyShell;

public:
  using Kernel = SExtruImplementation;

  VECCORE_ATT_HOST_DEVICE
  UnplacedSExtruVolume(int nvertices, Precision *x, Precision *y, Precision lowerz, Precision upperz)
      : fPolyShell(nvertices, x, y, lowerz, upperz)
  {
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  UnplacedSExtruVolume(UnplacedSExtruVolume const &other) : fPolyShell(other.fPolyShell) { ComputeBBox(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::sextruded; }

  VECCORE_ATT_HOST_DEVICE
  PolygonalShell const &GetStruct() const { return fPolyShell; }

  Precision Capacity() const override { return fPolyShell.fPolygon.Area() * (fPolyShell.fUpperZ - fPolyShell.fLowerZ); }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fPolyShell.SurfaceArea() + 2. * fPolyShell.fPolygon.Area(); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override { fPolyShell.Extent(aMin, aMax); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid(false);
    normal = SExtruImplementation::NormalKernel(fPolyShell, p, valid);
    return valid;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedSExtruVolume>::SizeOf(); }
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
