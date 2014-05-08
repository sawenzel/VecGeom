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

} // End namespace vecgeom_cuda

namespace vecgeom {

double ToInBenchmarker::RunCuda(
    Precision *const posX, Precision *const posY,
    Precision *const posZ, Precision *const dirX, 
    Precision *const dirY, Precision *const dirZ,
    Precision *const distances) const {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;

  if (fVerbose > 0) printf("Running CUDA benchmark...");

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
  vecgeom_cuda::SOA3D<Precision> positions(posX, posY, posZ, fPointCount);
  vecgeom_cuda::SOA3D<Precision> directions(dirX, dirY, dirZ, fPointCount);
  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);
  vecgeom_cuda::Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::DistanceToInBenchmarkCudaKernel<<<launch.grid_size,
                                                      launch.block_size>>>(
        *v, positions, directions, fPointCount, distances
      );
    }
  }
  const double elapsed = timer.Stop();

  if (fVerbose > 0) printf(" Finished in %fs.\n", elapsed);

  return elapsed;
}

} // End namespace vecgeom