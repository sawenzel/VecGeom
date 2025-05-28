/// \file CudaManager.cu
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/management/CudaManager.h"
#include "VecGeom/base/Assert.h"

#include <stdio.h>

#include "VecGeom/backend/cuda/Backend.h"

namespace vecgeom {
inline namespace cuda {

__global__ void InitDeviceCompactPlacedVolBufferPtrCudaKernel(void *gpu_ptr)
{
  // gpu_ptr is some pointer on the device that was allocated by some other means
  globaldevicegeomdata::gCompactPlacedVolBuffer = (vecgeom::cuda::VPlacedVolume *)gpu_ptr;
}

void InitDeviceCompactPlacedVolBufferPtr(void *gpu_ptr)
{
  InitDeviceCompactPlacedVolBufferPtrCudaKernel<<<1, 1>>>(gpu_ptr);
  VECGEOM_DEVICE_API_CALL(GetLastError());
}

__global__ void InitDeviceLogicalVolumesPtrCudaKernel(void *gpu_ptr)
{
  // gpu_ptr is some pointer on the device that was allocated by some other means
  globaldevicegeomdata::gDeviceLogicalVolumes = (vecgeom::cuda::LogicalVolume *)gpu_ptr;
}

void InitDeviceLogicalVolumesPtr(void *gpu_ptr)
{
  InitDeviceLogicalVolumesPtrCudaKernel<<<1, 1>>>(gpu_ptr);
  VECGEOM_DEVICE_API_CALL(GetLastError());
}

__global__ void InitDeviceNavIndexPtrCudaKernel(void *gpu_ptr, int maxdepth)
{
  // gpu_ptr is some pointer on the device that was allocated by some other means
  globaldevicegeomdata::gNavIndex = (NavIndex_t *)gpu_ptr;
  globaldevicegeomdata::gMaxDepth = maxdepth;
}

void InitDeviceNavIndexPtr(void *gpu_ptr, int maxdepth)
{
  InitDeviceNavIndexPtrCudaKernel<<<1, 1>>>(gpu_ptr, maxdepth);
}

__global__ void CudaManagerPrintGeometryKernel(vecgeom::cuda::VPlacedVolume const *const world)
{
  printf("Geometry loaded on GPU:\n");
  world->PrintContent();
}

void CudaManagerPrintGeometry(vecgeom::cuda::VPlacedVolume const *const world)
{
  CudaManagerPrintGeometryKernel<<<1, 1>>>(world);
  VECGEOM_DEVICE_API_SYMBOL(GetLastError)();
  VECGEOM_DEVICE_API_CALL(DeviceSynchronize());
}
} // namespace cuda
} // End namespace vecgeom
