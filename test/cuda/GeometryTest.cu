// Author: Stephan Hageboeck, CERN, 2021

#include "GeometryTest.h"

#include <cstdio>
#include <err.h>

__managed__ std::size_t g_volumesVisited;
__device__ struct VolumeData {
  vecgeom::cuda::VPlacedVolume const * vol;
  unsigned int depth;
} volumeStack[10000];

__global__ void kernel_visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume *volume, GeometryInfo *geoData,
                                           const std::size_t nGeoData)
{
  g_volumesVisited = 0;
  auto stackp = volumeStack;
  *(stackp++) = {volume, 0};

  while (stackp > volumeStack) {
    auto const current = *(--stackp);

    VECGEOM_ASSERT(g_volumesVisited < nGeoData);
    geoData[g_volumesVisited++] = GeometryInfo{current.depth, *current.vol};

    // We push backwards in order to visit the first daughter first
    for (int i = current.vol->GetDaughters().size() - 1; i >= 0; --i) {
      auto daughter = current.vol->GetDaughters()[i];
      *stackp++ = VolumeData{daughter, current.depth + 1};
      VECGEOM_ASSERT(stackp - volumeStack < sizeof(volumeStack) / sizeof(VolumeData) && "Volume stack size exhausted");
    }
  }
}

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume *volume, std::size_t maxElem)
{
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Cuda error before visiting device geometry: '%s'", cudaGetErrorString(err));
  }

  GeometryInfo *geoDataGPU;
  cudaMalloc(&geoDataGPU, maxElem * sizeof(GeometryInfo));

  kernel_visitDeviceGeometry<<<1, 1>>>(volume, geoDataGPU, maxElem);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Visiting device geometry failed with '%s'", cudaGetErrorString(err));
  }

  std::vector<GeometryInfo> geoDataCPU(maxElem);
  cudaMemcpy(geoDataCPU.data(), geoDataGPU, maxElem * sizeof(GeometryInfo), cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Retrieving device geometry data failed with '%s'", cudaGetErrorString(err));
  }

  cudaFree(geoDataGPU);

  geoDataCPU.resize(g_volumesVisited);
  printf(" %zu visited.\n", g_volumesVisited);

  return geoDataCPU;
}
