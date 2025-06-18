/// @file unplaced_root_volume.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDROOTVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDROOTVOLUME_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/UnplacedVolume.h"
#include <TGeoShape.h>
#include <TGeoBBox.h>

namespace vecgeom {

class UnplacedRootVolume : public VUnplacedVolume {

private:
  UnplacedRootVolume(const UnplacedRootVolume &);            // Not implemented
  UnplacedRootVolume &operator=(const UnplacedRootVolume &); // Not implemented

  TGeoShape const *fRootShape;

public:
  UnplacedRootVolume(TGeoShape const *const rootShape) : fRootShape(rootShape) {}

  virtual ~UnplacedRootVolume() {}

  VECGEOM_FORCE_INLINE
  TGeoShape const *GetRootShape() const { return fRootShape; }

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const final
  {
    auto const box    = (TGeoBBox *)fRootShape;
    auto const origin = box->GetOrigin();
    aMin.Set(origin[0] - box->GetDX(), origin[1] - box->GetDY(), origin[2] - box->GetDZ());
    aMin.Set(origin[0] + box->GetDX(), origin[1] + box->GetDY(), origin[2] + box->GetDZ());
  }

  bool Contains(Vector3D<Precision> const &p) const override { return fRootShape->Contains(&Vector3D<double>(p)[0]); }

  EnumInside Inside(Vector3D<Precision> const &point) const override
  {
    return Contains(point) ? static_cast<EnumInside>(EInside::kInside) : static_cast<EnumInside>(EInside::kOutside);
  }

  Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                         const Precision stepMax) const override
  {
    return GetRootShape()->DistFromOutside(&Vector3D<double>(position)[0], &Vector3D<double>(direction)[0], 3);
  }

  Precision DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                          const Precision stepMax) const override
  {
    return GetRootShape()->DistFromInside(&Vector3D<double>(position)[0], &Vector3D<double>(direction)[0], 3);
  }

  Precision SafetyToOut(Vector3D<Precision> const &position) const override
  {
    return GetRootShape()->Safety(&Vector3D<double>(position)[0], true);
  }

  Precision SafetyToIn(Vector3D<Precision> const &position) const override
  {
    return GetRootShape()->Safety(&Vector3D<double>(position)[0], false);
  }

  Precision Capacity() const override { return GetRootShape()->Capacity(); }

  Precision SurfaceArea() const override { return 0.; /*GetRootShape()->SurfaceArea();*/ }

  VECGEOM_FORCE_INLINE
  int MemorySize() const override { return sizeof(*this); }

  virtual bool Normal(Vector3D<Precision> const &pos, Vector3D<Precision> &normal) const override { return false; }

  void Print() const override;

  void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return 0; /* DevicePtr<cuda::UnplacedRootVolume>::SizeOf(); */ }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

private:
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           VPlacedVolume *const placement = NULL) const override;
};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDROOTVOLUME_H_
