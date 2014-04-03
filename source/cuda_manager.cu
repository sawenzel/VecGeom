#include "management/cuda_manager.h"

#include <stdio.h>

#include "backend/cuda/backend.h"

namespace vecgeom {

__global__
void CudaManagerPrintGeometryKernel(
    vecgeom_cuda::VPlacedVolume const *const world) {
  printf("Geometry loaded on GPU:\n");
  world->PrintContent();
}

void CudaManagerPrintGeometry(VPlacedVolume const *const world) {
  CudaManagerPrintGeometryKernel<<<1, 1>>>(
    reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(world)
  );
  CudaAssertError();
  cudaDeviceSynchronize();
}

} // End namespace vecgeom