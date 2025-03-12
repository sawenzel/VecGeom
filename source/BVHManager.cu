/// \file BVHManager.cu
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/backend/cuda/Interface.h"
#include <VecGeom/navigation/BVHNavigatorV.h>
#include <VecGeom/navigation/BVHSafetyEstimator.h>

using vecgeom::cxx::CudaCheckError;

namespace vecgeom {
inline namespace cuda {
template <typename Real_t>
inline BVH<Real_t> *AllocateDeviceBVHBuffer(size_t n)
{
  BVH<Real_t> *ptr = nullptr;
  CudaCheckError(cudaMalloc((void **)&ptr, n * sizeof(BVH<Real_t>)));
  CudaCheckError(cudaMemcpyToSymbol(dBVH<Real_t>, &ptr, sizeof(ptr)));
  CudaCheckError(cudaDeviceSynchronize());
  return ptr;
}

template <typename Real_t>
inline BVH<Real_t> *GetDeviceBVHBuffer()
{
  BVH<Real_t> *ptr = nullptr;

  CudaCheckError(cudaMemcpyFromSymbol(&ptr, dBVH<Real_t>, sizeof(ptr)));
  return ptr;
}

template <typename Real_t>
inline void FreeDeviceBVHBuffer()
{
  CudaCheckError(cudaFree(GetDeviceBVHBuffer<Real_t>()));
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
VECCORE_ATT_DEVICE VNavigator *BVHNavigatorV<false>::Instance()
{
  if (gBVHNavigatorV == nullptr) gBVHNavigatorV = new BVHNavigatorV();
  return gBVHNavigatorV;
}

template BVH<float> *AllocateDeviceBVHBuffer<float>(size_t);
template BVH<double> *AllocateDeviceBVHBuffer<double>(size_t);

template BVH<float> *GetDeviceBVHBuffer<float>();
template BVH<double> *GetDeviceBVHBuffer<double>();

template void FreeDeviceBVHBuffer<float>();
template void FreeDeviceBVHBuffer<double>();

} // namespace cuda
} // namespace vecgeom
