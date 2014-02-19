#include <algorithm>
#include <vector>
#include "management/cuda_manager.h"

namespace vecgeom {

VLogicalVolume const* CUDAManager::CopyToGPU(VLogicalVolume const &top_volume) {

  logical_volumes.clear();
  unplaced_volumes.clear();
  placed_volumes.clear();
  matrices.clear();

  // NYI
  return NULL;

}

void CUDAManager::ScanGeometry(VLogicalVolume const *const volume) {

  if (!VectorContains(logical_volumes, volume)) {
    logical_volumes.push_back(volume);
  }
  if (!VectorContains(unplaced_volumes, &volume->unplaced_volume())) {
    unplaced_volumes.push_back(&volume->unplaced_volume());
  }
  for (Iterator<VPlacedVolume const*> i = volume->daughters().begin();
       i != volume->daughters().end();
       ++i) {
    if (!VectorContains(matrices, &((*i)->matrix()))) {
      matrices.push_back(&(*i)->matrix());
    }
    ScanGeometry(&(*i)->logical_volume());
  }

}

template <typename VectorType>
bool CUDAManager::VectorContains(std::vector<VectorType> const &vec,
                                 const VectorType element) {
  if (std::find(vec.begin(), vec.end(), element) != vec.end()) return true;
  return false;
}

void CUDAManager::PrintContent() const {
  std::cout << "-- Logical volumes:\n";
  for (std::vector<VLogicalVolume const*>::const_iterator i =
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {
    std::cout << "  " << (**i) << std::endl;
  }
  std::cout << "-- Unplaced volumes:\n";
  for (std::vector<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {
    std::cout << "  " << (**i) << std::endl;
  }
  std::cout << "-- Placed volumes:\n";
  for (std::vector<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {
    std::cout << "  " << (**i) << std::endl;
  }
  std::cout << "-- Transformation matrices:\n";
  for (std::vector<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {
    std::cout << "  " << (**i) << std::endl;
  }
}

} // End namespace vecgeom