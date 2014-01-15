#include <iostream> 
#include "LibraryVc.h"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
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

  Stopwatch timer;
  timer.Start();
  box.DistanceToIn<kVc>(points, directions, step_max, output);
  timer.Stop();
  
  std::cout << "Vc benchmark for " << n_points << " points finished in "
            << timer.Elapsed() << "s.\n";

  int hit = 0;
  for (int i = 0; i < n_points; ++i) {
    if (output[i] < kInfinity) {
      hit++;
    }
  }

  std::cout << double(hit)/double(n_points)
            << " hit something.\n";

  return 0;
}