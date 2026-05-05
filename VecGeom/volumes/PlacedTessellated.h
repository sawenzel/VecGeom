// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the data structure for the tessellated shape.
/// \file volumes/PlacedTessellated.h
/// \author Mihaela Gheata (CERN/ISS)

#ifndef VECGEOM_VOLUMES_PLACEDTESSELLATED_H_
#define VECGEOM_VOLUMES_PLACEDTESSELLATED_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/TessellatedImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedTessellated.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTessellated;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTessellated);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class representing a placed tessellated volume.
class PlacedTessellated : public PlacedVolumeImplHelper<UnplacedTessellated, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedTessellated, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  PlacedTessellated(char const *const label, LogicalVolume const *const logicalVolume,
                    Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  PlacedTessellated(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedTessellated("", logicalVolume, transformation)
  {
  }
#else
  /// CUDA version of constructor
  __device__ PlacedTessellated(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTessellated() {}

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

  /// Getter for the UnplacedTessellated
  VECCORE_ATT_HOST_DEVICE
  UnplacedTessellated const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTessellated const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  /** @brief Memory size in bytes */
  int MemorySize() const override { return sizeof(*this); }

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
