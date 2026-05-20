/// \file PlacedRootVolume.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "UnplacedRootVolume.h"

#include "TGeoShape.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedRootVolume;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedRootVolume);

inline namespace cxx {

class PlacedRootVolume : public VPlacedVolume {

private:
  PlacedRootVolume(const PlacedRootVolume &);            // Not implemented
  PlacedRootVolume &operator=(const PlacedRootVolume &); // Not implemented

public:
  PlacedRootVolume(char const *const label, TGeoShape const *const rootShape, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation);

  PlacedRootVolume(TGeoShape const *const rootShape, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation);

  virtual ~PlacedRootVolume() {}

  TGeoShape const *GetRootShape() const { return ((UnplacedRootVolume *)GetUnplacedVolume())->GetRootShape(); }

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &) const override;

  VECGEOM_FORCE_INLINE
  virtual bool Contains(Vector3D<Precision> const &point) const override;

  VECGEOM_FORCE_INLINE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const override;

  VECGEOM_FORCE_INLINE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override;

  VECGEOM_FORCE_INLINE
  virtual EnumInside Inside(Vector3D<Precision> const &point) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max,
                                 SurfaceHitView<Precision> *hit_info = nullptr) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                  Precision const stepMax,
                                  SurfaceHitView<Precision> *hit_info = nullptr) const override;

  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                        Precision const stepMax,
                                        SurfaceHitView<Precision> *hit_info = nullptr) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const override;

  virtual void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  virtual Precision Capacity() override;

  virtual Precision SurfaceArea() const override
  {
    throw std::runtime_error("unimplemented function called");
    return -1.;
  }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override
  {
    VECGEOM_VALIDATE(0, << "Attempted to perform conversion on unsupported ROOT volume.");
    return nullptr;
  }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override
  {
    VECGEOM_VALIDATE(0, << "Attempted to perform conversion on unsupported ROOT volume.");
    return nullptr;
  }
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return 0; /* return DevicePtr<cuda::PlacedRootVolume>::SizeOf(); */ }
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform,
                                                   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const override;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform) const override;
  virtual void CopyManyToGpu(std::vector<VPlacedVolume const *> const &host_volumes,
                             std::vector<DevicePtr<cuda::LogicalVolume>> const &logical_volumes,
                             std::vector<DevicePtr<cuda::Transformation3D>> const &transforms,
                             std::vector<DevicePtr<cuda::VPlacedVolume>> const &in_gpu_ptrs) const override
  {
  }
#endif
};

bool PlacedRootVolume::Contains(Vector3D<Precision> const &point) const
{
  const Vector3D<Precision> local = GetTransformation()->Transform(point);
  return UnplacedContains(local);
}

bool PlacedRootVolume::Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const
{
  localPoint = GetTransformation()->Transform(point);
  return UnplacedContains(localPoint);
}

bool PlacedRootVolume::UnplacedContains(Vector3D<Precision> const &point) const
{
  return GetRootShape()->Contains(&Vector3D<double>(point)[0]);
}

EnumInside PlacedRootVolume::Inside(Vector3D<Precision> const &point) const
{
  const Vector3D<Precision> local = GetTransformation()->Transform(point);
  return (UnplacedContains(local)) ? static_cast<EnumInside>(EInside::kInside)
                                   : static_cast<EnumInside>(EInside::kOutside);
}

Precision PlacedRootVolume::DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                         const Precision stepMax, SurfaceHitView<Precision> *hit_info) const
{
  if (hit_info) hit_info->Clear();
  Vector3D<double> positionLocal  = GetTransformation()->Transform(position);
  Vector3D<double> directionLocal = GetTransformation()->TransformDirection(direction);
  return GetRootShape()->DistFromOutside(&positionLocal[0], &directionLocal[0], 3);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                          const Precision stepMax, SurfaceHitView<Precision> *hit_info) const
{
  if (hit_info) hit_info->Clear();
  return GetRootShape()->DistFromInside(&Vector3D<double>(position)[0], &Vector3D<double>(direction)[0], 3);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::PlacedDistanceToOut(Vector3D<Precision> const &position,
                                                Vector3D<Precision> const &direction, const Precision stepMax,
                                                SurfaceHitView<Precision> *hit_info) const
{
  if (hit_info) hit_info->Clear();
  Vector3D<double> positionLocal  = GetTransformation()->Transform(position);
  Vector3D<double> directionLocal = GetTransformation()->TransformDirection(direction);
  return GetRootShape()->DistFromInside(&positionLocal[0], &directionLocal[0], 3);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::SafetyToOut(Vector3D<Precision> const &position) const
{
  Vector3D<double> position_local = GetTransformation()->Transform(position);
  return GetRootShape()->Safety(&position_local[0], true);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::SafetyToIn(Vector3D<Precision> const &position) const
{
  Vector3D<double> position_local = GetTransformation()->Transform(position);
  return GetRootShape()->Safety(&position_local[0], false);
}
} // namespace cxx
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_
