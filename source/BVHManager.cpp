/// \file BVHManager.cpp
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/Logger.h"

#include <vector>

namespace vecgeom {
#ifdef VECCORE_CUDA
inline
#endif
    namespace cuda {

BVH *AllocateDeviceBVHBuffer(size_t n);
BVH *GetDeviceBVHBuffer();
void FreeDeviceBVHBuffer();

} // namespace cuda

inline namespace VECGEOM_IMPL_NAMESPACE {
void BVHManager::Init()
{
  std::vector<LogicalVolume const *> lvols;
  GeoManager::Instance().GetAllLogicalVolumes(lvols);
  // There may be volumes not used in the hierarchy, so the maximum index may be larger
  hBVH.resize(GeoManager::Instance().GetLogicalVolumesMap().size());
  for (auto logical_volume : lvols)
    hBVH[logical_volume->id()] = logical_volume->GetDaughters().size() > 0 ? new BVH(*logical_volume) : nullptr;
}

cuda::BVH const *BVHManager::DeviceInit()
{
#ifdef VECGEOM_CUDA_INTERFACE
  int n = hBVH.size();

  cuda::BVH *ptr = vecgeom::cuda::AllocateDeviceBVHBuffer(n);

  for (int id = 0; id < n; ++id) {
    if (!hBVH[id]) continue;

    hBVH[id]->CopyToGpu(&reinterpret_cast<cxx::BVH *>(ptr)[id]);
  }
  return ptr;
#else
  VECGEOM_LOG(error) << "Cannot initialize BVH device: CUDA is not configured";
  return nullptr;
#endif
}

cuda::BVH const *BVHManager::GetDeviceBVH()
{
#ifdef VECGEOM_CUDA_INTERFACE
  return cuda::GetDeviceBVHBuffer();
#else
  return nullptr;
#endif
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
