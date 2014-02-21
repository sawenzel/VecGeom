#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include <iostream>
#include "base/types.h"
#include "base/vector.h"
#include "volumes/unplaced_volume.h"

namespace vecgeom {

typedef VPlacedVolume const* Daughter;

class LogicalVolume {

private:

  VUnplacedVolume const &unplaced_volume_;
  Container<Daughter> *daughters_;

  friend class CudaManager;

public:

  LogicalVolume(VUnplacedVolume const &unplaced_volume__)
      : unplaced_volume_(unplaced_volume__) {
    daughters_ = new Vector<Daughter>();
  }

  /**
   * Constructor for building geometry on the GPU.
   */
  LogicalVolume(VUnplacedVolume const *const unplaced_ptr,
                Container<Daughter> *const daughters_ptr)
      : unplaced_volume_(*unplaced_ptr), daughters_(daughters_ptr) {}

  ~LogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const& unplaced_volume() const { return unplaced_volume_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Container<Daughter> const& daughters() const {
    return *daughters_;
  }

  void PlaceDaughter(LogicalVolume const &volume,
                     TransformationMatrix const &matrix);

  friend std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol);

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_