/// \file PlacedParaboloid.h

#ifndef VECGEOM_VOLUMES_PLACEDPARABOLOID_H_
#define VECGEOM_VOLUMES_PLACEDPARABOLOID_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedParaboloid.h"

namespace VECGEOM_NAMESPACE {

class PlacedParaboloid : public VPlacedVolume {

public:

  typedef UnplacedParaboloid UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedParaboloid(char const *const label,
                   LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedParaboloid(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : PlacedParaboloid("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedParaboloid(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedParaboloid() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParaboloid const* GetUnplacedVolume() const {
    return static_cast<UnplacedParaboloid const *>(
        logical_volume()->unplaced_volume());
  }

  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDPARABOLOID_H_