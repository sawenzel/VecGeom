/// @file PlacedExtruded.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDEXTRUDED_H_
#define VECGEOM_VOLUMES_PLACEDEXTRUDED_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedExtruded.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedExtruded;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedExtruded);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedExtruded : public PlacedVolumeImplHelper<UnplacedExtruded, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedExtruded, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedExtruded(char const *const label, LogicalVolume const *const logicalVolume,
                 Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedExtruded(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedExtruded("", logicalVolume, transformation)
  {
  }
#else
  __device__ PlacedExtruded(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                            const int id, const int copy_no, const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedExtruded() {}

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

  VECCORE_ATT_HOST_DEVICE
  UnplacedExtruded const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedExtruded const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

#ifndef VECCORE_CUDA
  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { return GetUnplacedVolume()->ConvertToRoot(GetName()); }
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4() const override { return GetUnplacedVolume()->ConvertToGeant4(GetName()); }
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
