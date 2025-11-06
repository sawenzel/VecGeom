/// \file BVHManager.cu
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/backend/cuda/Interface.h"
#include <VecGeom/navigation/BVHSafetyEstimator.h>

#include "VecGeom/base/Assert.h"

namespace vecgeom {
inline namespace cuda {
template <typename Real_t>
inline BVH<Real_t> *AllocateDeviceBVHBuffer(size_t n)
{
  BVH<Real_t> *ptr = nullptr;
  VECGEOM_DEVICE_API_CALL(Malloc((void **)&ptr, n * sizeof(BVH<Real_t>)));
  VECGEOM_DEVICE_API_CALL(MemcpyToSymbol(dBVH<Real_t>, &ptr, sizeof(ptr)));
  VECGEOM_DEVICE_API_CALL(DeviceSynchronize());
  return ptr;
}

template <typename Real_t>
inline BVH<Real_t> *GetDeviceBVHBuffer()
{
  BVH<Real_t> *ptr = nullptr;

  VECGEOM_DEVICE_API_CALL(MemcpyFromSymbol(&ptr, dBVH<Real_t>, sizeof(ptr)));
  return ptr;
}

template <typename Real_t>
inline void FreeDeviceBVHBuffer()
{
  VECGEOM_DEVICE_API_CALL(Free(GetDeviceBVHBuffer<Real_t>()));
}

// Temporary hack (used already in LogicalVolume.cpp) implementing the Instance functionality
// on device for BVHSafetyEstimator and BVHNavigatorV in the absence of the corresponding
// implementation files
VECCORE_ATT_DEVICE
BVHSafetyEstimator *gBVHSafetyEstimator = nullptr;

VECCORE_ATT_DEVICE
VSafetyEstimator *BVHSafetyEstimator::Instance()
{
  if (gBVHSafetyEstimator == nullptr) gBVHSafetyEstimator = new BVHSafetyEstimator();
  return gBVHSafetyEstimator;
}

template BVH<float> *AllocateDeviceBVHBuffer<float>(size_t);
template BVH<double> *AllocateDeviceBVHBuffer<double>(size_t);

template BVH<float> *GetDeviceBVHBuffer<float>();
template BVH<double> *GetDeviceBVHBuffer<double>();

template void FreeDeviceBVHBuffer<float>();
template void FreeDeviceBVHBuffer<double>();

} // namespace cuda
} // namespace vecgeom
