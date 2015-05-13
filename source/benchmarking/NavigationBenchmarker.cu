
#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/SOA3D.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"

#ifdef VECGEOM_CUDA
#include "base/Stopwatch.h"
#include "backend/cuda/Backend.h"
#include "management/CudaManager.h"
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {
inline namespace cuda {

__global__
void NavigationKernel(void* gpu_ptr /* a pointer to buffer of current navigation states */,
                      void* gpu_out_ptr /* a pointer to buffer for next states */,
                      int depth, VPlacedVolume const *const volume,
                      const SOA3D<Precision> positions, const SOA3D<Precision> directions,
                      Precision const * pSteps,  const int n,  Precision *const steps) {

  using vecgeom::cuda::NavigationState;
  using vecgeom::cuda::NavStatePool;
  SimpleNavigator nav;
  double step;

  unsigned tid = ThreadIndex();
  while (tid < n) {

    //.. get the navigationstate for this thread/lane
    NavigationState *inState  = reinterpret_cast<NavigationState*>((char*)gpu_ptr + tid*NavigationState::SizeOf(depth));
    NavigationState *outState = reinterpret_cast<NavigationState*>((char*)gpu_out_ptr + tid*NavigationState::SizeOf(depth));

    //.. do the actual navigation on the GPU
    // nav.LocatePoint(volume, positions[tid], *inState, true);
    nav.FindNextBoundaryAndStep(positions[tid], directions[tid], *inState, *outState, pSteps[tid], step);
    steps[tid] = step;

    // repeat
    tid += ThreadOffset();
  }
}

} // end of namespace cuda

// Should this function be moved to NavigationBenchmarker.cpp?
  Precision runNavigationCuda(
    void *gpu_ptr, void *gpu_out_ptr, int depth,
    const cxx::VPlacedVolume *const volume, unsigned npoints,
    Precision const *const posX, Precision const *const posY, Precision const *const posZ,
    Precision const *const dirX, Precision const *const dirY, Precision const *const dirZ,
    Precision const *const maxSteps, Precision *const propSteps )
  {

   // transfer geometry to GPU
   using CudaVolume = cuda::VPlacedVolume const*;
   using CudaSOA3D  = cuda::SOA3D<Precision>;
   using cxx::CudaManager;

   // build a list of GPU volume pointers - needed?

   // copy points to the GPU
   cxx::DevicePtr<Precision> posXGpu; posXGpu.Allocate(npoints);
   cxx::DevicePtr<Precision> posYGpu; posYGpu.Allocate(npoints);
   cxx::DevicePtr<Precision> posZGpu; posZGpu.Allocate(npoints);
   posXGpu.ToDevice(posX, npoints);
   posYGpu.ToDevice(posY, npoints);
   posZGpu.ToDevice(posZ, npoints);
   CudaSOA3D positionGpu = CudaSOA3D(posXGpu, posYGpu, posZGpu, npoints);

   // copy directions to the GPU
   cxx::DevicePtr<Precision> dirXGpu; dirXGpu.Allocate(npoints);
   cxx::DevicePtr<Precision> dirYGpu; dirYGpu.Allocate(npoints);
   cxx::DevicePtr<Precision> dirZGpu; dirZGpu.Allocate(npoints);
   dirXGpu.ToDevice(dirX, npoints);
   dirYGpu.ToDevice(dirY, npoints);
   dirZGpu.ToDevice(dirZ, npoints);
   CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, npoints);

   cxx::DevicePtr<Precision> maxStepsGpu;  maxStepsGpu.Allocate(npoints);
   maxStepsGpu.ToDevice(maxSteps, npoints);

   // allocate space for kernel output
   Precision* propStepsGpu = cxx::AllocateOnGpu<Precision>(npoints*sizeof(Precision));

   // launch kernel in GPU
   vecgeom::cuda::LaunchParameters launch(npoints);
   vecgeom::cuda::Stopwatch timer;

   // printf("GPU warm-up:  <<<1,32>>>\n");
   // vecgeom::cuda::NavigationKernel<<< 1, 32>>>(
   //   gpu_ptr, gpu_out_ptr, depth, CudaManager::Instance().world_gpu(),
   //   positionGpu, directionGpu, maxStepsGpu, 32, propStepsGpu );
   // cudaDeviceSynchronize();

   printf("GPU configuration:  <<<%i,%i>>>\n", launch.grid_size.x, launch.block_size.x);

   timer.Start();
   vecgeom::cuda::NavigationKernel<<< launch.grid_size, launch.block_size>>>(
     gpu_ptr, gpu_out_ptr, depth, CudaManager::Instance().world_gpu(),
     positionGpu, directionGpu, maxStepsGpu, npoints, propStepsGpu );
   cudaDeviceSynchronize();
   Precision elapsedCuda = timer.Stop();

   cxx::CopyFromGpu(propStepsGpu, propSteps, npoints*sizeof(Precision));
   for(size_t i=0; i<10; ++i) {
     std::cout<<"NavBenchmarker.cu: propSteps["<< i <<"] = "<< propSteps[i] << std::endl;
   }

   cxx::FreeFromGpu(propStepsGpu);
   posXGpu.Deallocate();
   posYGpu.Deallocate();
   posZGpu.Deallocate();
   dirXGpu.Deallocate();
   dirYGpu.Deallocate();
   dirZGpu.Deallocate();

   // compare steps from navigator with the ones above
   std::cout<<"GPU navigation time: "<< 1000.*elapsedCuda <<" ms\n";
   return elapsedCuda;
}

} // global namespace
