#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include <iostream>
#include <string>
#include "base/global.h"
#include "base/vector.h"
#include "volumes/unplaced_volume.h"

namespace vecgeom {

typedef VPlacedVolume const* Daughter;

class LogicalVolume {

private:

  VUnplacedVolume const *unplaced_volume_;
  Vector<Daughter> *daughters_;

  friend class CudaManager;

public:

  LogicalVolume(VUnplacedVolume const *const unplaced_volume__)
      : unplaced_volume_(unplaced_volume__) {
    daughters_ = new Vector<Daughter>();
  }

  VECGEOM_CUDA_HEADER_DEVICE
  LogicalVolume(VUnplacedVolume const *const unplaced_volume,
                Vector<Daughter> *daughters)
      : unplaced_volume_(unplaced_volume), daughters_(daughters) {}

  ~LogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const { return unplaced_volume_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const& daughters() const { return *daughters_; }

  VPlacedVolume* Place(TransformationMatrix const *const matrix) const;

  VPlacedVolume* Place() const;

  void PlaceDaughter(LogicalVolume const *const volume,
                     TransformationMatrix const *const matrix);

  VECGEOM_CUDA_HEADER_BOTH
  int CountVolumes() const;

  /**
   * Recursively prints contained logical volumes.
   */
  VECGEOM_CUDA_HEADER_BOTH
  void PrintContent(const int depth = 0) const;

  friend std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol);

  #ifdef VECGEOM_CUDA
  LogicalVolume* CopyToGpu(VUnplacedVolume const *const unplaced_volume,
                           Vector<Daughter> *daughters) const;
  LogicalVolume* CopyToGpu(VUnplacedVolume const *const unplaced_volume,
                           Vector<Daughter> *daughters,
                           LogicalVolume *const gpu_ptr) const;
  #endif

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_