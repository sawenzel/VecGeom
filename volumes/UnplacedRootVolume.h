/// @file unplaced_root_volume.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDROOTVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDROOTVOLUME_H_

#include "base/Global.h"

#include "volumes/UnplacedVolume.h"

class TGeoShape;

namespace vecgeom {

class UnplacedRootVolume : public VUnplacedVolume {

private:
   UnplacedRootVolume(const UnplacedRootVolume&); // Not implemented
   UnplacedRootVolume& operator=(const UnplacedRootVolume&); // Not implemented

  TGeoShape const *fRootShape;

public:

  UnplacedRootVolume(TGeoShape const *const rootShape)
      : fRootShape(rootShape) {}

  virtual ~UnplacedRootVolume() {}

  VECGEOM_INLINE
  TGeoShape const* GetRootShape() const { return fRootShape; }

  VECGEOM_INLINE
  virtual int memory_size() const { return sizeof(*this); }

  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return 0; /* DevicePtr<cuda::UnplacedRootVolume>::SizeOf(); */ }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const;

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDROOTVOLUME_H_
