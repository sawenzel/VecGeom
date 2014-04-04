#include "cpugpu.h"

#include "backend/cuda/backend.h"
#include "base/soa3d.h"
#include "management/cuda_manager.h"
#include "navigation/simple_navigator.h"
#include "navigation/navigationstate.h"
#include "volumes/placed_volume.h"

using namespace vecgeom_cuda;

__global__
void LocatePointsKernel(
    vecgeom_cuda::VPlacedVolume const *const world,
    SimpleNavigator const *const navigator,
    NavigationState *const paths, const vecgeom_cuda::SOA3D<Precision> points,
    const int n, int *const output) {
  const int i = ThreadIndex();
  if (i >= n) return;
  output[i] = navigator->LocatePoint(world, points[i], paths[i], true)->id();
}

__global__
void LocatePointsInitialize(
    SimpleNavigator *const navigator, NavigationState *const states,
    const int depth) {
  const int i = ThreadIndex();
  new(&states[i]) NavigationState(depth);
  if (i == 0) new(navigator) SimpleNavigator();
}

void LocatePointsGpu(Precision *const x, Precision *const y, Precision *const z,
                     const unsigned size, const int depth, int *const output) {

  vecgeom_cuda::SOA3D<Precision> points(x, y, z, size);
  Precision *const x_gpu = AllocateOnGpu<Precision>(sizeof(Precision)*size);
  Precision *const y_gpu = AllocateOnGpu<Precision>(sizeof(Precision)*size);
  Precision *const z_gpu = AllocateOnGpu<Precision>(sizeof(Precision)*size);
  points.CopyToGpu(x_gpu, y_gpu, z_gpu, size);

  SimpleNavigator *const navigator = AllocateOnGpu<SimpleNavigator>();
  NavigationState *const paths = AllocateOnGpu<NavigationState>(
    size*sizeof(NavigationState)
  );
  LaunchParameters launch(size);
  LocatePointsInitialize<<<launch.grid_size, launch.block_size>>>(
    navigator, paths, depth
  );
  int *const output_gpu = AllocateOnGpu<int>(size*sizeof(int));

  LocatePointsKernel<<<launch.grid_size, launch.block_size>>>(
    CudaManager::Instance().world_gpu(),
    navigator,
    paths,
    points,
    size,
    output_gpu
  );

  CopyFromGpu(output_gpu, output, size*sizeof(int));

  FreeFromGpu(navigator);
  FreeFromGpu(paths);
  FreeFromGpu(output_gpu);
  FreeFromGpu(x_gpu);
  FreeFromGpu(y_gpu);
  FreeFromGpu(z_gpu);
}