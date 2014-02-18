#include <vector>
#include "management/cuda_manager.h"

namespace vecgeom {

VLogicalVolume const* CUDAManager::CopyToGPU(VLogicalVolume const &top_volume) {

  unplaced_volumes.clear();
  logical_volumes.clear();
  placed_volumes.clear();
  matrices.clear();

  // NYI

}

void CUDAManager::ScanGeometry(VLogicalVolume const *const volume) {

  // NYI

}

} // End namespace vecgeom