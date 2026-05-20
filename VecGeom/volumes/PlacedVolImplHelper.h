#pragma once

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include <algorithm>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(struct, PlacedVolumeImplHelper, class, Arg1, class, Arg2);

inline namespace VECGEOM_IMPL_NAMESPACE {

// A helper template class to automatically implement (wire)
// interfaces from PlacedVolume using kernel functions and functionality
// from the unplaced shapes
template <class UnplacedShape_t, class BaseVol = VPlacedVolume>
struct PlacedVolumeImplHelper : public BaseVol {

  using Struct_t = typename UnplacedShape_t::UnplacedStruct_t;

public:
  using BaseVol::BaseVol;
  using BaseVol::GetLogicalVolume;
  using BaseVol::PrintType;

  // destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedVolumeImplHelper() {}

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  UnplacedShape_t const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedShape_t const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  Struct_t const *GetUnplacedStruct() const { return static_cast<Struct_t const *>(&GetUnplacedVolume()->GetStruct()); }

#if !defined(VECCORE_CUDA)
  virtual Precision Capacity() override { return const_cast<UnplacedShape_t *>(GetUnplacedVolume())->Capacity(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    VPlacedVolume::Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return VPlacedVolume::Normal(point, normal);
  }
#endif

  virtual Precision SurfaceArea() const override { return GetUnplacedVolume()->SurfaceArea(); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::Contains(point);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfLength) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::DistanceToOut(point, direction, stepMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  const Precision stepMax, SurfaceHitView<Precision> *hit_info) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::DistanceToOut(point, direction, stepMax, hit_info);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::SafetyToOut(point);
  }

}; // End class PlacedVolumeImplHelper
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
