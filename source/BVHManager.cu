/// \file BVHManager.cu
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/backend/cuda/Interface.h"
#include <VecGeom/navigation/BVHNavigatorV.h>
#include <VecGeom/navigation/BVHSafetyEstimator.h>

using vecgeom::cxx::CudaCheckError;

namespace vecgeom {
inline namespace cuda {
void *AllocateDeviceBVHBuffer(size_t n)
{
  BVH *ptr = nullptr;
  CudaCheckError(cudaMalloc((void **)&ptr, n * sizeof(BVH)));
  CudaCheckError(cudaMemcpyToSymbol(dBVH, &ptr, sizeof(ptr)));
  CudaCheckError(cudaDeviceSynchronize());
  return (void*) ptr;
}

void FreeDeviceBVHBuffer()
{
  void *ptr = nullptr;

  CudaCheckError(cudaMemcpyFromSymbol(&ptr, dBVH, sizeof(ptr)));

  if (ptr)
    CudaCheckError(cudaFree(ptr));
}

// Temporary hack (used already in LogicalVolume.cpp) implementing the Instance functionality
// on device for BVHSafetyEstimator and BVHNavigatorV in the absence of the corresponding
// implementation files
VECCORE_ATT_DEVICE
BVHSafetyEstimator *gBVHSafetyEstimator = nullptr;

VECCORE_ATT_DEVICE
VNavigator *gBVHNavigatorV = nullptr;

VECCORE_ATT_DEVICE
VSafetyEstimator *BVHSafetyEstimator::Instance()
{
  if (gBVHSafetyEstimator == nullptr) gBVHSafetyEstimator = new BVHSafetyEstimator();
  return gBVHSafetyEstimator;
}

template <>
VECCORE_ATT_DEVICE
VNavigator *BVHNavigatorV<false>::Instance()
{
  if (gBVHNavigatorV == nullptr) gBVHNavigatorV = new BVHNavigatorV();
  return gBVHNavigatorV;
}

} // namespace cuda
} // namespace vecgeom
