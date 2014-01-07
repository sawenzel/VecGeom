#include <iostream>
#include <thrust/random.h>
#include "LibraryCuda.cuh"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
}

double LaunchCuda(Box const * const box, SOA3D<double> const &points,
                  bool *output) {

  SOA3D<double> points_dev = points.CopyToGPU();
  bool *output_dev = AllocateOnGPU<bool>(points.size());

  Stopwatch timer;
  timer.Start();
  box->Contains<kCuda>(points_dev, output_dev);
  timer.Stop();

  CopyFromGPU(output_dev, output, points.size());
  points_dev.FreeFromGPU();
  cudaFree(output_dev);

  return timer.Elapsed();
}

int main(void) {

  const int n_points = 1<<19;

  // Populate some points
  SOA3D<double> points(n_points);
  for (int i = 0; i < n_points; ++i) {
    (points.Memory(0))[i] = random(-10, 10);
    (points.Memory(1))[i] = random(-10, 10);
    (points.Memory(2))[i] = random(-10, 10);
  }

  // Create a box
  const Vector3D<double> box_dim(5., 5., 5.);
  const TransMatrix<double> *box_pos = new TransMatrix<double>();
  const Box box(box_dim, box_pos);

  // Output vector
  bool *output = (bool*) _mm_malloc(sizeof(bool)*n_points,
                                    kAlignmentBoundary);

  // Launch kernel
  double elapsed = LaunchCuda(&box, points, output);
  
  std::cout << "CUDA benchmark for " << n_points << " points finished in "
            << elapsed << "s.\n";

  int inside = 0;
  for (int i = 0; i < n_points; ++i) if (output[i]) inside++;

  std::cout << double(inside)/double(n_points)
            << " were inside the box.\n";
}