/// \file CudaManager.cu
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/CudaManager.h"

#include <stdio.h>

#include "backend/cuda/Backend.h"

namespace vecgeom {
inline namespace cuda {

__global__
void CudaManagerPrintGeometryKernel(
    vecgeom::cuda::VPlacedVolume const *const world) {
  printf("Geometry loaded on GPU:\n");
  world->PrintContent();
}

void CudaManagerPrintGeometry(vecgeom::cuda::VPlacedVolume const *const world) {
  CudaManagerPrintGeometryKernel<<<1, 1>>>(world);
  cxx::CudaAssertError();
  cudaDeviceSynchronize();
}

} } // End namespace vecgeom