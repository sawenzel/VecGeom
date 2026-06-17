/// @file PlacedHalfSpace.h
/// @brief Placed-volume wrapper for Boolean half-space operands.

#ifndef VECGEOM_VOLUMES_PLACEDHALFSPACE_H_
#define VECGEOM_VOLUMES_PLACEDHALFSPACE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedHalfSpace.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/HalfSpaceImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedHalfSpace;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedHalfSpace);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// @brief Placed representation of an infinite half-space.
/// @details Half-spaces are intended as Boolean CSG operands. They can be
/// placed inside Boolean trees, but geometry placement guards reject direct
/// use as world or daughter volumes.
class PlacedHalfSpace : public PlacedVolumeImplHelper<UnplacedHalfSpace, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedHalfSpace, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  using Base::Base;
  /// @brief Construct a named placed half-space.
  /// @param label Placed volume label.
  /// @param logicalVolume Logical volume owning the unplaced half-space.
  /// @param transformation Transform from the mother frame to the half-space frame.
  PlacedHalfSpace(char const *const label, LogicalVolume const *const logicalVolume,
                  Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  /// @brief Construct an unnamed placed half-space.
  PlacedHalfSpace(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedHalfSpace("", logicalVolume, transformation)
  {
  }
#else
  /// @brief Device-side constructor used by CUDA geometry copying.
  VECCORE_ATT_DEVICE
  PlacedHalfSpace(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation, const int id,
                  const int copy_no, const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedHalfSpace() {}

  /// @brief Return one point on the local limiting plane.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetPoint() const { return GetUnplacedVolume()->GetPoint(); }

  /// @brief Return the local unit normal pointing outside the half-space.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetNormal() const { return GetUnplacedVolume()->GetNormal(); }

  /// @brief Print the concrete placed half-space type to stdout.
  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  /// @brief Print the concrete placed half-space type to a stream.
  virtual void PrintType(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  /// @brief Convert to the unspecialized placed half-space representation.
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  /// @brief Convert to ROOT's `TGeoHalfSpace` when ROOT support is enabled.
  virtual TGeoShape const *ConvertToRoot() const override { return GetUnplacedVolume()->ConvertToRoot(GetName()); }
#endif
#ifdef VECGEOM_GEANT4
  /// @brief Return null because Geant4 has no half-space solid.
  virtual G4VSolid const *ConvertToGeant4() const override { return nullptr; }
#endif
#endif // VECCORE_CUDA
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDHALFSPACE_H_
