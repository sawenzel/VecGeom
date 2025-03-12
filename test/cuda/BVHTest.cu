#include "VecGeom/management/BVHManager.h"

#include <err.h>

using namespace vecgeom;

__global__ void check_device_bvh_kernel(int id)
{
  if (auto const *bvh = BVHManager::GetBVH(id)) bvh->Print();
}

void check_device_bvh(int id)
{
  check_device_bvh_kernel<<<1, 1>>>(id);
  if (cudaDeviceSynchronize() != cudaSuccess) warnx("Invalid BVH for volume with id = %d\n", id);
}
