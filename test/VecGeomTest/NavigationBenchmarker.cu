
#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/NavigationState.h"

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/backend/cuda/Backend.h"
#include "VecGeom/management/CudaManager.h"
#include "VecGeom/backend/cuda/Interface.h"
#endif

namespace vecgeom {
inline namespace cuda {

__global__ void NavigationKernel(void *gpu_ptr /* a pointer to buffer of current navigation states */,
                                 void *gpu_out_ptr /* a pointer to buffer for next states */,
                                 VPlacedVolume const *const volume, const SOA3D<Precision> positions,
                                 const SOA3D<Precision> directions, Precision const *pSteps, const int n,
                                 Precision *const steps)
{

  using vecgeom::cuda::NavigationState;
  auto nav = NewSimpleNavigator<>::Instance(); // pointer to a navigator
  Precision step;
  auto *inStates  = reinterpret_cast<NavigationState *>(gpu_ptr);
  auto *outStates = reinterpret_cast<NavigationState *>(gpu_out_ptr);

  unsigned tid = ThreadIndex();
  while (tid < n) {

    //.. get the navigationstate for this thread/lane
    NavigationState *inState  = &inStates[tid];
    NavigationState *outState = &outStates[tid];

    //.. do the actual navigation on the GPU
    // nav.LocatePoint(volume, positions[tid], *inState, true);
    nav->FindNextBoundaryAndStep(positions[tid], directions[tid], *inState, *outState, pSteps[tid], step);
    steps[tid] = step;

    // repeat
    tid += ThreadOffset();
  }
}

} // end of namespace cuda

// Should this function be moved to NavigationBenchmarker.cpp?
Precision runNavigationCuda(void *gpu_ptr, void *gpu_out_ptr, const cxx::VPlacedVolume *const volume, unsigned npoints,
                            Precision const *const posX, Precision const *const posY, Precision const *const posZ,
                            Precision const *const dirX, Precision const *const dirY, Precision const *const dirZ,
                            Precision const *const maxSteps, Precision *const propSteps)
{
  // transfer geometry to GPU
  using CudaVolume = cuda::VPlacedVolume const *;
  using CudaSOA3D  = cuda::SOA3D<Precision>;
  using cxx::CudaManager;

  // build a list of GPU volume pointers - needed?

  // copy points to the GPU
  cxx::DevicePtr<Precision> posXGpu;
  posXGpu.Allocate(npoints);
  cxx::DevicePtr<Precision> posYGpu;
  posYGpu.Allocate(npoints);
  cxx::DevicePtr<Precision> posZGpu;
  posZGpu.Allocate(npoints);
  posXGpu.ToDevice(posX, npoints);
  posYGpu.ToDevice(posY, npoints);
  posZGpu.ToDevice(posZ, npoints);
  CudaSOA3D positionGpu = CudaSOA3D(posXGpu, posYGpu, posZGpu, npoints);

  // copy directions to the GPU
  cxx::DevicePtr<Precision> dirXGpu;
  dirXGpu.Allocate(npoints);
  cxx::DevicePtr<Precision> dirYGpu;
  dirYGpu.Allocate(npoints);
  cxx::DevicePtr<Precision> dirZGpu;
  dirZGpu.Allocate(npoints);
  dirXGpu.ToDevice(dirX, npoints);
  dirYGpu.ToDevice(dirY, npoints);
  dirZGpu.ToDevice(dirZ, npoints);
  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, npoints);

  cxx::DevicePtr<Precision> maxStepsGpu;
  maxStepsGpu.Allocate(npoints);
  maxStepsGpu.ToDevice(maxSteps, npoints);

  // allocate space for kernel output
  Precision *propStepsGpu = cxx::AllocateOnGpu<Precision>(npoints * sizeof(Precision));

  // launch kernel in GPU
  vecgeom::cuda::LaunchParameters launch(npoints);
  vecgeom::cuda::Stopwatch timer;

  timer.Start();
  vecgeom::cuda::NavigationKernel<<<1, 1>>>(gpu_ptr, gpu_out_ptr, CudaManager::Instance().world_gpu(), positionGpu,
                                            directionGpu, maxStepsGpu, 1, propStepsGpu);
  cudaDeviceSynchronize();
  Precision elapsedWarmup = timer.Stop();
  printf("GPU config <<<1,1>>> - warm-up time: %f ms\n", 1000. * elapsedWarmup);

  timer.Start();
  vecgeom::cuda::NavigationKernel<<<launch.grid_size, launch.block_size>>>(
      gpu_ptr, gpu_out_ptr, CudaManager::Instance().world_gpu(), positionGpu, directionGpu, maxStepsGpu, npoints,
      propStepsGpu);
  cudaDeviceSynchronize();
  Precision elapsedCuda = timer.Stop();
  printf("GPU config <<<%i,%i>>> - navigation time: %f ms\n", launch.grid_size.x, launch.block_size.x,
         1000. * elapsedCuda);

  cxx::CopyFromGpu(propStepsGpu, propSteps, npoints * sizeof(Precision));

  // cleanup
  cxx::FreeFromGpu(propStepsGpu);
  posXGpu.Deallocate();
  posYGpu.Deallocate();
  posZGpu.Deallocate();
  dirXGpu.Deallocate();
  dirYGpu.Deallocate();
  dirZGpu.Deallocate();
  maxStepsGpu.Deallocate();

  return elapsedCuda;
}

} // namespace vecgeom
