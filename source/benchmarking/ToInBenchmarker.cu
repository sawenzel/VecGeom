/// @file ToInBenchmarker.cu
/// @author Johannes de Fine Licht

#include "benchmarking/ToInBenchmarker.h"

#include "base/stopwatch.h"
#include "backend/cuda/backend.h"
#include "management/cuda_manager.h"

namespace vecgeom_cuda {

__global__
void DistanceToInBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const SOA3D<Precision> directions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->DistanceToIn(positions[i], directions[i]);
}

__global__
void SafetyToInBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->SafetyToIn(positions[i]);
}

} // End namespace vecgeom_cuda

namespace vecgeom {

void ToInBenchmarker::RunCuda(
    Precision *const posX, Precision *const posY,
    Precision *const posZ, Precision *const dirX, 
    Precision *const dirY, Precision *const dirZ,
    Precision *const distances, Precision *const safeties) const {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;
  typedef vecgeom_cuda::SOA3D<Precision> CudaSOA3D;

  double elapsedDistance;
  double elapsedSafety;

  printf("Running CUDA benchmark...");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->specialized())
      )
    );
  }

  Precision *posXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirX, dirXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirY, dirYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirZ, dirZGpu, fPointCount*sizeof(Precision));

  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);
  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, fPointCount);

  Precision *distancesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                     *fPointCount);
  Precision *safetiesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                    *fPointCount);

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);
  vecgeom_cuda::Stopwatch timer;
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::DistanceToInBenchmarkCudaKernel<<<launch.grid_size,
                                                      launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  elapsedDistance = timer.Stop();
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::SafetyToInBenchmarkCudaKernel<<<launch.grid_size,
                                                    launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  elapsedSafety = timer.Stop();

  printf(" Finished in %fs/%fs.\n", elapsedDistance, elapsedSafety);

  CopyFromGpu(distancesGpu, distances, fPointCount*sizeof(Precision));
  CopyFromGpu(safetiesGpu, safeties, fPointCount*sizeof(Precision));

  FreeFromGpu(distancesGpu);
  FreeFromGpu(safetiesGpu);
  FreeFromGpu(posXGpu);
  FreeFromGpu(posYGpu);
  FreeFromGpu(posZGpu);
  FreeFromGpu(dirXGpu);
  FreeFromGpu(dirYGpu);
  FreeFromGpu(dirZGpu);
}

} // End namespace vecgeom