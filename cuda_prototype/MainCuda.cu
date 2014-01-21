#include <iostream>
#include <fstream>
#include "LibraryCuda.cuh"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
}

double LaunchCuda(BoxParameters const * const box,
                  TransMatrix<double> const * const matrix,
                  SOA3D<double> const &points,
                  SOA3D<double> const &directions, double const *step_max,
                  double *output) {

  SOA3D<double> points_dev = points.CopyToGPU();
  SOA3D<double> directions_dev = directions.CopyToGPU();
  double *output_dev = AllocateOnGPU<double>(points.size());
  double *step_max_dev = AllocateOnGPU<double>(points.size());
  CopyToGPU(step_max, step_max_dev, points.size());
  TransMatrix<CudaFloat> *trans_matrix_dev =
      AllocateOnGPU<TransMatrix<CudaFloat> >(1);
  TransMatrix<CudaFloat> converted(*matrix);
  CopyToGPU(&converted, trans_matrix_dev, 1);
  const Box box_cuda(box, trans_matrix_dev);

  CheckCudaError();

  Stopwatch timer;
  timer.Start();
  box_cuda.DistanceToIn(points_dev, directions_dev, step_max_dev, output_dev);
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

  Stopwatch timer, timer_total;
  timer_total.Start();
  
  const int n_points = 1<<21;

  const TransMatrix<double> origin;
  TransMatrix<double> pos;
  pos.SetTranslation(2.93, 1.30, -4.05);
  Box world(new BoxParameters(Vector3D<double>(10., 10., 10.)), &origin);
  BoxParameters box_params(Vector3D<double>(2., 1., 4.));
  Box box(&box_params, &pos);
  world.AddDaughter(&box);

  SOA3D<double> points(n_points);
  SOA3D<double> directions(n_points);
  timer.Start();
  std::cerr << "Loading points and directions...";
  std::string filename("io/single_box.in");
  std::ifstream filestream;
  filestream.open(filename.c_str());
  for (int i = 0; i < n_points; ++i) {
    std::string line;
    std::getline(filestream, line);
    const int semicolon = line.find(";");
    points.Set(i, Vector3D<double>(line.substr(0, semicolon)));
    directions.Set(i, Vector3D<double>(line.substr(semicolon+1)));
  }
  filestream.close();
  std::cerr << " Done in " << timer.Stop() << "s.\n";

  double *step_max = (double*) _mm_malloc(sizeof(double)*n_points,
                                          kAlignmentBoundary);
  double *output = (double*) _mm_malloc(sizeof(double)*n_points,
                                        kAlignmentBoundary);
  timer.Start();
  for (int i = 0; i < n_points; ++i) {
    step_max[i] = kInfinity;
  }
  std::cout << "Max step array initialized in " << timer.Stop() << "s.\n";

  timer.Start();
  double inner = LaunchCuda(&box_params, &pos, points, directions, step_max,
                            output);
  
  std::cout << "CUDA benchmark for " << n_points << " points finished in "
            << inner << "s (" << timer.Stop() << "s with memory overhead)\n";

  int hit = 0;
  filename = "io/single_box.out.cuda";
  std::ofstream outstream;
  outstream.open(filename.c_str());
  for (int i = 0; i < n_points; ++i) {
    outstream << output[i] << std::endl;
    if (output[i] < kInfinity) {
      hit++;
    }
  }
  outstream.close();
  std::cout << "Hits counted in " << timer.Stop() << "s.\n";

  std::cout << hit << " / " << n_points << " (" << double(hit)/double(n_points)
            << " of total) hit something.\n";

  std::cout << "Total binary execution time: " << timer_total.Stop() << "s.\n";

  return 0;
}