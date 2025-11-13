/// \file BVHManager.cpp
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/Logger.h"

#include <vector>

namespace vecgeom {
#ifdef VECCORE_CUDA
inline
#endif
    namespace cuda {

template <typename Real_t>
BVH<Real_t> *AllocateDeviceBVHBuffer(size_t n);
template <typename Real_t>
BVH<Real_t> *GetDeviceBVHBuffer();
void FreeDeviceBVHBuffer();

} // namespace cuda

inline namespace VECGEOM_IMPL_NAMESPACE {
void BVHManager::Init()
{
  std::vector<LogicalVolume const *> lvols;
  GeoManager::Instance().GetAllLogicalVolumes(lvols);
  // There may be volumes not used in the hierarchy, so the maximum index may be larger
  hBVH<Real_t>.resize(GeoManager::Instance().GetLogicalVolumesMap().size());
  for (auto logical_volume : lvols) {
    int n{0};
    auto ptrAABB = ABBoxManager<Precision>::Instance().GetABBoxes(logical_volume, n);
    hBVH<Real_t>[logical_volume->id()] =
        logical_volume->GetDaughters().size() > 0 ? new BVH<Real_t>(*logical_volume, ptrAABB, n) : nullptr;
  }
}

cuda::BVH<BVHManager::Real_t> const *BVHManager::DeviceInit()
{
#ifdef VECGEOM_CUDA_INTERFACE
  int n = hBVH<Real_t>.size();

  cuda::BVH<Real_t> *ptr = vecgeom::cuda::AllocateDeviceBVHBuffer<Real_t>(n);

  for (int id = 0; id < n; ++id) {
    if (!hBVH<Real_t>[id]) continue;

    hBVH<Real_t>[id]->CopyToGpu(&reinterpret_cast<cxx::BVH<Real_t> *>(ptr)[id]);
  }
  return ptr;
#else
  VECGEOM_LOG(error) << "Cannot initialize BVH device: CUDA is not configured";
  return nullptr;
#endif
}

cuda::BVH<BVHManager::Real_t> const *BVHManager::GetDeviceBVH()
{
#ifdef VECGEOM_CUDA_INTERFACE
  return cuda::GetDeviceBVHBuffer<Real_t>();
#else
  return nullptr;
#endif
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
