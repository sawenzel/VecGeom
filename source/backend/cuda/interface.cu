/**
 * @file interface.cu
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <cassert>
#include <iostream>
#include <stdio.h>
 
#include "base/soa3d.h"
#include "base/aos3d.h"
#include "backend/cuda/backend.h"
#include "backend/cuda/interface.h"
#include "navigation/navigationstate.h"
#include "navigation/simple_navigator.h"
#include "volumes/placed_volume.h"
#include "volumes/logical_volume.h"

namespace vecgeom {

cudaError_t CudaCheckError(const cudaError_t err) {
  if (err != cudaSuccess) {
    std::cout << "CUDA reported error with message: \""
              << cudaGetErrorString(err) << "\"\n";
  }
  return err;
}

cudaError_t CudaCheckError() {
  return CudaCheckError(cudaGetLastError());
}

void CudaAssertError(const cudaError_t err) {
  assert(CudaCheckError(err) == cudaSuccess);
}

void CudaAssertError() {
  CudaAssertError(cudaGetLastError());
}

cudaError_t CudaMalloc(void** ptr, unsigned size) {
  return cudaMalloc(ptr, size);
}

cudaError_t CudaCopyToDevice(void* tgt, void const* src, unsigned size) {
  return cudaMemcpy(tgt, src, size, cudaMemcpyHostToDevice);
}

cudaError_t CudaCopyFromDevice(void* tgt, void const* src, unsigned size) {
  return cudaMemcpy(tgt, src, size, cudaMemcpyDeviceToHost);
}

cudaError_t CudaFree(void* ptr) {
  return cudaFree(ptr);
}

// Class specific functions

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

template <typename TrackContainer>
__global__
void CudaManagerLocatePointsKernel(
    vecgeom_cuda::VPlacedVolume const *const world,
    vecgeom_cuda::SimpleNavigator const *const navigator,
    vecgeom_cuda::NavigationState *const paths,
    TrackContainer const *const points,
    int *const output) {
  const int i =vecgeom_cuda::ThreadIndex();
  output[i] =
      navigator->LocatePoint(world, (*points)[i], paths[i], true)->id();
}

__global__
void CudaManagerLocatePointsInitialize(
    vecgeom_cuda::SimpleNavigator *const navigator,
    vecgeom_cuda::NavigationState *const states, const int depth) {
  const int i =vecgeom_cuda::ThreadIndex();
  new(&states[i]) vecgeom_cuda::NavigationState(depth);
  if (i == 0) new(navigator) vecgeom_cuda::SimpleNavigator();
}

template <typename TrackContainer>
void CudaManagerLocatePointsTemplate(VPlacedVolume const *const world,
                                     TrackContainer const *const points,
                                     const int n, const int depth,
                                     int *const output) {

  vecgeom_cuda::SimpleNavigator *const navigator =
      AllocateOnGpu<vecgeom_cuda::SimpleNavigator>();
  vecgeom_cuda::NavigationState *const paths =
      AllocateOnGpu<vecgeom_cuda::NavigationState>(
        n*sizeof(vecgeom_cuda::NavigationState)
      );
  vecgeom_cuda::LaunchParameters launch(n);
  CudaManagerLocatePointsInitialize<<<launch.grid_size, launch.block_size>>>(
    navigator, paths, depth
  );
  int *const output_gpu = AllocateOnGpu<int>(n*sizeof(int));

  CudaManagerLocatePointsKernel<<<launch.grid_size, launch.block_size>>>(
    reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(world),
    navigator,
    paths,
    points,
    output_gpu
  );

  CopyFromGpu(output_gpu, output, n*sizeof(int));

  FreeFromGpu(navigator);
  FreeFromGpu(paths);
  FreeFromGpu(output_gpu);
}

void CudaManagerLocatePoints(VPlacedVolume const *const world,
                             SOA3D<Precision> const *const points,
                             const int n, const int depth, int *const output) {
  CudaManagerLocatePointsTemplate(
    world,
    reinterpret_cast<vecgeom_cuda::SOA3D<Precision> const*>(points),
    n,
    depth,
    output
  );
}

void CudaManagerLocatePoints(VPlacedVolume const *const world,
                             AOS3D<Precision> const *const points,
                             const int n, const int depth, int *const output) {
  CudaManagerLocatePointsTemplate(
    world,
    reinterpret_cast<vecgeom_cuda::AOS3D<Precision> const*>(points),
    n,
    depth,
    output
  );
}

} // End namespace vecgeom