#include <iostream>
#include <thrust/random.h>
#include "LibraryCuda.cuh"
#include "LauncherCuda.cuh"

CudaFloat random(const CudaFloat low, const CudaFloat high) {
  return low + static_cast<CudaFloat>(rand()) /
               static_cast<CudaFloat>(RAND_MAX / (high - low));
}

int main(void) {

  const int n_points = 1<<20;

  // Populate some points
  SOA3D_CUDA_Float points(n_points);
  for (int i = 0; i < n_points; ++i) {
    (points.Memory(0))[i] = random(-10, 10);
    (points.Memory(1))[i] = random(-10, 10);
    (points.Memory(2))[i] = random(-10, 10);
  }

  // Create a "box"
  const Vector3D<CudaScalar> box_pos(0., 0., 0.);
  const Vector3D<CudaScalar> box_dim(5., 5., 5.);

  // Output vector
  CudaBool* output = (CudaBool*) _mm_malloc(sizeof(CudaBool)*n_points,
                                            kAlignmentBoundary);

  // Launch kernel
  Stopwatch timer;
  timer.Start();
  LauncherCuda::Contains(box_pos, box_dim, points, output);
  timer.Stop();
  
  std::cout << "CUDA benchmark for " << n_points << " points finished in "
            << timer.Elapsed() << "s.\n";

  int inside = 0;
  for (int i = 0; i < n_points; ++i) {
    if (output[i]) inside++;
  }

  std::cout << inside << " / " << n_points << " were inside the box.\n";
}