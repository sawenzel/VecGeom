// Author: Stephan Hageboeck, CERN, 2021

#include "GeometryTest.h"

#include <cstdio>
#include <err.h>

__managed__ std::size_t g_volumesVisited;
struct VolumeData {
  vecgeom::cuda::VPlacedVolume const *vol;
  unsigned int depth;
};

__global__ void kernel_visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume *volume, GeometryInfo *geoData,
                                           const std::size_t nGeoData, VolumeData *volumeStack,
                                           const std::size_t stackCapacity)
{
  g_volumesVisited = 0;
  VECGEOM_ASSERT(stackCapacity > 0);
  auto stackp   = volumeStack;
  auto stackEnd = volumeStack + stackCapacity;
  *(stackp++)   = {volume, 0};

  while (stackp > volumeStack) {
    auto const current = *(--stackp);

    VECGEOM_ASSERT(g_volumesVisited < nGeoData);
    geoData[g_volumesVisited++] = GeometryInfo{current.depth, *current.vol};

    // We push backwards in order to visit the first daughter first
    for (int i = static_cast<int>(current.vol->GetDaughters().size()) - 1; i >= 0; --i) {
      auto daughter = current.vol->GetDaughters()[i];
      VECGEOM_ASSERT(stackp < stackEnd && "Volume stack size exhausted");
      *stackp++ = VolumeData{daughter, current.depth + 1};
    }
  }
}

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume *volume, std::size_t maxElem,
                                              std::size_t stackCapacity)
{
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Cuda error before visiting device geometry: '%s'", cudaGetErrorString(err));
  }

  GeometryInfo *geoDataGPU;
  err = cudaMalloc(&geoDataGPU, maxElem * sizeof(GeometryInfo));
  if (err != cudaSuccess) {
    errx(2, "Allocating device geometry data failed with '%s'", cudaGetErrorString(err));
  }

  VolumeData *volumeStackGPU;
  err = cudaMalloc(&volumeStackGPU, stackCapacity * sizeof(VolumeData));
  if (err != cudaSuccess) {
    cudaFree(geoDataGPU);
    errx(2, "Allocating device traversal stack failed with '%s'", cudaGetErrorString(err));
  }

  kernel_visitDeviceGeometry<<<1, 1>>>(volume, geoDataGPU, maxElem, volumeStackGPU, stackCapacity);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(volumeStackGPU);
    cudaFree(geoDataGPU);
    errx(2, "Visiting device geometry failed with '%s'", cudaGetErrorString(err));
  }

  std::vector<GeometryInfo> geoDataCPU(maxElem);
  cudaMemcpy(geoDataCPU.data(), geoDataGPU, maxElem * sizeof(GeometryInfo), cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(volumeStackGPU);
    cudaFree(geoDataGPU);
    errx(2, "Retrieving device geometry data failed with '%s'", cudaGetErrorString(err));
  }

  cudaFree(volumeStackGPU);
  cudaFree(geoDataGPU);

  geoDataCPU.resize(g_volumesVisited);
  printf(" %zu visited.\n", g_volumesVisited);

  return geoDataCPU;
}
