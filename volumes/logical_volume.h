#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include <iostream>
#include <string>
#include "base/types.h"
#include "base/vector.h"
#include "volumes/unplaced_volume.h"

namespace vecgeom {

typedef VPlacedVolume const* Daughter;

class LogicalVolume {

private:

  VUnplacedVolume const *unplaced_volume_;
  Container<Daughter> *daughters_;

  friend class CudaManager;

  /**
   * Constructor used for copying to the GPU by CudaManager.
   */
  LogicalVolume(VUnplacedVolume const *const unplaced_volume__,
                Container<Daughter> *daughters__)
      : unplaced_volume_(unplaced_volume__), daughters_(daughters__) {}

public:

  LogicalVolume(VUnplacedVolume const *const unplaced_volume__)
      : unplaced_volume_(unplaced_volume__) {
    daughters_ = new Vector<Daughter>();
  }

  ~LogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const { return unplaced_volume_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Container<Daughter> const& daughters() const {
    return *daughters_;
  }

  void PlaceDaughter(LogicalVolume const *const volume,
                     TransformationMatrix const *const matrix);

  /**
   * Recursively prints contained logical volumes.
   */
  void PrintContent(std::string prefix = "") const;

  friend std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol);

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_