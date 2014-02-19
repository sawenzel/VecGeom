#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/types.h"
#include "base/transformation_matrix.h"
#include "management/geo_manager.h"
#include "volumes/logical_volume.h"

namespace vecgeom {

class VPlacedVolume {

private:

  int id;

protected:

  VLogicalVolume const &logical_volume_;
  TransformationMatrix const &matrix_;

public:

  VPlacedVolume(VLogicalVolume const &logical_volume__,
                TransformationMatrix const &matrix__)
      : logical_volume_(logical_volume__), matrix_(matrix__) {
    id = GeoManager::Instance().RegisterVolume(this);
  }

  ~VPlacedVolume() {
    GeoManager::Instance().DeregisterVolume(this);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VLogicalVolume const &logical_volume() const {
    return logical_volume_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const& unplaced_volume() const {
    return logical_volume().unplaced_volume();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  TransformationMatrix const& matrix() const {
    return matrix_;
  }

  VECGEOM_CUDA_HEADER_HOST
  friend std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol);

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_]