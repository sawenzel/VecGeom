#ifndef VECGEOM_MANAGEMENT_CUDAMANAGER_H_
#define VECGEOM_MANAGEMENT_CUDAMANAGER_H_

#include "volumes/box.h"

namespace vecgeom {

class CUDAManager {

private:

  std::vector<VUnplacedVolume const*> unplaced_volumes;
  std::vector<VLogicalVolume const*> logical_volumes;
  std::vector<VPlacedVolume const*> placed_volumes;
  std::vector<TransformationMatrix const*> matrices;

public:

  static CUDAManager& Instance() {
    static CUDAManager instance;
    return instance;
  }

  VLogicalVolume const* CopyToGPU(VLogicalVolume const &top_volume);

  void PrintContent() const;

private:

  CUDAManager() {}
  CUDAManager(CUDAManager const&);
  CUDAManager& operator=(CUDAManager const&);

  void ScanGeometry(VLogicalVolume const *const volume);

  template <typename VectorType>
  bool VectorContains(std::vector<VectorType> const &vec,
                      const VectorType element);

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_CUDAMANAGER_H_