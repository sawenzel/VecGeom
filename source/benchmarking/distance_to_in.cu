#include "backend/cuda/backend.h"
#include "benchmarking/distance_to_in.h"

namespace vecgeom_cuda {

__global__
void DistanceToInBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    SOA3D<Precision> const *const positions,
    SOA3D<Precision> const *const directions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->DistanceToIn(positions[i], directions[i]);
}

} // End namespace vecgeom_cuda

namespace vecgeom {

BenchmarkResult DistanceToInBenchmarker::RunCuda(
    Precision *const distances) const {
  if (verbose()) std::cout << "Running CUDA benchmark...";
  LaunchParameters launch = LaunchParameters(n_points_);
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    DistanceToInBenchmarkCudaKernel<<<launch.grid_size, launch.block_size>>>(
      reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(volume),
      reinterpret_cast<vecgeom_cuda::SOA3D<Precision> const*>(positions),
      reinterpret_cast<vecgeom_cuda::SOA3D<Precision> const*>(directions),
      n,
      distances
    );
  }
  const double elapsed = timer.Stop();
  if (verbose()) std::cout << "Finished in " << elapsed << "s.\n";
}

} // End namespace vecgeom