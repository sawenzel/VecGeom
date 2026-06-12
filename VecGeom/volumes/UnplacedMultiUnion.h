#ifndef VECGEOM_VOLUMES_UNPLACEDMULTIUNION_H_
#define VECGEOM_VOLUMES_UNPLACEDMULTIUNION_H_
/**
 @brief Class representing an unplaced union of multiple placed solids, possibly overlapping.
 Implemented based on the class G4MultipleUnion
 @author mihaela.gheata@cern.ch
*/

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "MultiUnionStruct.h"
#include "VecGeom/volumes/kernel/MultiUnionImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedMultiUnion);
VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedMultiUnion;);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedMultiUnion : public UnplacedVolumeImplHelper<MultiUnionImplementation>, public AlignedBase {

protected:
  mutable MultiUnionStruct fMultiUnion; ///< Structure storing multi-union parameters

public:
  // the constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedMultiUnion() : fMultiUnion()
  {
    fGlobalConvexity = false;
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddNode(VUnplacedVolume const *volume, Transformation3D const &transform)
  {
    LogicalVolume *lvol = new LogicalVolume(volume);
    VPlacedVolume *pvol = lvol->Place(new Transformation3D(transform));
    fMultiUnion.AddNode(pvol);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddNode(VPlacedVolume const *volume) { fMultiUnion.AddNode(volume); }

  VECCORE_ATT_HOST_DEVICE
  void Close()
  {
    fMultiUnion.Close();
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VPlacedVolume const *GetNode(size_t i) const { return fMultiUnion.fVolumes[i]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::multiunion; }

  VECCORE_ATT_HOST_DEVICE
  MultiUnionStruct const &GetStruct() const { return fMultiUnion; }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  Precision Capacity() const override;

  Precision SurfaceArea() const override;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNumberOfSolids() const { return fMultiUnion.fVolumes.size(); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    aMin = fMultiUnion.fMinExtent;
    aMax = fMultiUnion.fMaxExtent;
  }

  Vector3D<Precision> SamplePointOnSurface() const override;

  std::string GetEntityType() const { return "MultiUnion"; }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override {};

  virtual void Print(std::ostream & /*os*/) const override {};

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
#ifdef VECGEOM_CUDA_HYBRID2
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedMultiUnion>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#else
  virtual size_t DeviceSizeOf() const override { return 0; }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override { return DevicePtr<cuda::VUnplacedVolume>(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override
  {
    return DevicePtr<cuda::VUnplacedVolume>(gpu_ptr);
  }
#endif
#endif

private:
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const override;

}; // End class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_UNPLACEDMULTIUNION_H_ */
