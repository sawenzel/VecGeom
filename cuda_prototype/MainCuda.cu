#include <iostream>
#include <thrust/random.h>
#include "LibraryCuda.cuh"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
}

double LaunchCuda(Box * const box, SOA3D<double> const &points,
                  SOA3D<double> const &directions, double const *step_max,
                  double *output) {

  SOA3D<double> points_dev = points.CopyToGPU();
  SOA3D<double> directions_dev = points.CopyToGPU();
  double *output_dev = AllocateOnGPU<double>(points.size());
  double *step_max_dev = AllocateOnGPU<double>(points.size());
  CopyToGPU(step_max, step_max_dev, points.size());
  TransMatrix<float> *trans_matrix_dev = AllocateOnGPU<TransMatrix<float> >(1);
  TransMatrix<float> converted(*box->TransformationMatrix());
  CopyToGPU(&converted, trans_matrix_dev, 1);
  box->SetCudaMatrix(trans_matrix_dev);

  CheckCudaError();

  Stopwatch timer;
  timer.Start();
  box->DistanceToIn<kCuda>(points_dev, directions_dev, step_max_dev,
                           output_dev);
  timer.Stop();

  CopyFromGPU(output_dev, output, points.size());
  points_dev.FreeFromGPU();
  directions_dev.FreeFromGPU();
  cudaFree(output_dev);
  cudaFree(step_max_dev);
  cudaFree(trans_matrix_dev);

  CheckCudaError();

  return timer.Elapsed();
}

int main(void) {

  const int n_points = 1<<19;

  const TransMatrix<double> *origin = new TransMatrix<double>();
  TransMatrix<double> *pos = new TransMatrix<double>();
  pos->SetTranslation(2.93, 1.30, -4.05);
  Box world(Vector3D<double>(10., 10., 10.), origin);
  Box box(Vector3D<double>(2., 2., 2.), pos);
  world.AddDaughter(&box);

  SOA3D<double> points(n_points);
  SOA3D<double> directions(n_points);
  world.FillUncontainedPoints(points);
  world.FillBiasedDirections(points, 0.8, directions);

  double *step_max = (double*) _mm_malloc(sizeof(double)*n_points,
                                          kAlignmentBoundary);
  double *output = (double*) _mm_malloc(sizeof(double)*n_points,
                                        kAlignmentBoundary);
  for (int i = 0; i < n_points; ++i) {
    step_max[i] = kInfinity;
  }

  const double elapsed = LaunchCuda(&box, points, directions, step_max, output);
  
  std::cout << "CUDA benchmark for " << n_points << " points finished in "
            << elapsed << "s.\n";

  int hit = 0;
  for (int i = 0; i < n_points; ++i) {
    if (output[i] < kInfinity) hit++;
  }

  std::cout << double(hit)/double(n_points)
            << " hit something.\n";

  return 0;
}