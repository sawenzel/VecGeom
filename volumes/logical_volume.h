#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include "base/types.h"
#include "base/vector.h"
#include "volumes/unplaced_volume.h"

namespace vecgeom {

class VLogicalVolume {

private:

  VUnplacedVolume const &unplaced_volume_;
  Container<VPlacedVolume const*> *daughters;

  friend VPlacedVolume;

public:

  VECGEOM_CUDA_HEADER_HOST
  VLogicalVolume(VUnplacedVolume const &unplaced_volume__)
      : unplaced_volume_(unplaced_volume__) {
    daughters = new Vector<VPlacedVolume const*>();
  }

  VECGEOM_CUDA_HEADER_HOST
  ~VLogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const &unplaced_volume() const { return unplaced_volume_; }

  VECGEOM_CUDA_HEADER_HOST
  void PlaceDaughter(VLogicalVolume const &volume,
                     TransformationMatrix const &matrix);

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_