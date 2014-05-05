/// @file ShapeImplementationHelper.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_

#include "base/global.h"
#include "volumes/placed_volume.h"
#include "volumes/UnplacedParallelepiped.h"

namespace VECGEOM_NAMESPACE {

class PlacedParallelepiped : public VPlacedVolume {

private:

  UnplacedParallelepiped const *fUnplacedParallelepiped;

public:

  typedef UnplacedParallelepiped UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedParallelepiped(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedParallelepiped(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedParallelepiped("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedParallelepiped(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       const int id,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(logical_volume, transformation, id) {}

#endif

  virtual ~PlacedParallelepiped() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped const* GetUnplacedVolume() const {
    return fUnplacedParallelepiped;
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_