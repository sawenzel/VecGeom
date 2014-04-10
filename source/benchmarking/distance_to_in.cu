#include "benchmarking/distance_to_in.h"

#include "base/stopwatch.h"
#include "backend/cuda/backend.h"
#include "management/cuda_manager.h"

#include <iostream>

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

double DistanceToInBenchmarker::RunCuda(
    Precision *const pos_x, Precision *const pos_y,
    Precision *const pos_z, Precision *const dir_x, 
    Precision *const dir_y, Precision *const dir_z,
    Precision *const distances) const {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;

  if (verbose()) std::cout << "Running CUDA benchmark...";

  CudaManager::Instance().LoadGeometry(this->world());
  CudaManager::Instance().Synchronize();
  std::vector<CudaVolume> volumes_gpu;
  for (std::vector<VolumePointers>::const_iterator v = volumes_.begin();
       v != volumes_.end(); ++v) {
    volumes_gpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->specialized())
      )
    );
  }
  vecgeom_cuda::SOA3D<Precision> positions(pos_x, pos_y, pos_z, n_points_);
  vecgeom_cuda::SOA3D<Precision> directions(dir_x, dir_y, dir_z, n_points_);

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(n_points_);
  vecgeom_cuda::Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    for (std::vector<CudaVolume>::const_iterator v = volumes_gpu.begin();
         v != volumes_gpu.end(); ++v) {
      vecgeom_cuda::DistanceToInBenchmarkCudaKernel<<<launch.grid_size,
                                                      launch.block_size>>>(
        *v, positions, directions, n_points_, distances
      );
    }
  }
  const double elapsed = timer.Stop();

  if (verbose()) std::cout << "Finished in " << elapsed << "s.\n";

  return elapsed;
}

} // End namespace vecgeom