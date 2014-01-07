#include <iostream>
#include "LibraryVc.h"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
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

  // SOA3D_Vc_Float points(n_points);
  // for (int i = 0; i < points.vectors(); ++i) {
  //   points.Memory(0).vector(i) = VcFloat::Random() * 20. - 10.;
  //   points.Memory(1).vector(i) = VcFloat::Random() * 20. - 10.;
  //   points.Memory(2).vector(i) = VcFloat::Random() * 20. - 10.;
  // }

  // Create a box
  const Vector3D<double> box_dim(5., 5., 5.);
  const TransMatrix<double> *box_pos = new TransMatrix<double>();
  const Box box(box_dim, box_pos);

  // Output vector
  bool *output = (bool*) _mm_malloc(sizeof(bool)*n_points,
                                    kAlignmentBoundary);

  // Launch kernel
  Stopwatch timer;
  timer.Start();
  box.Contains<kVc>(points, output);
  timer.Stop();
  
  std::cout << "Vc benchmark for " << n_points << " points finished in "
            << timer.Elapsed() << "s.\n";

  int inside = 0;
  for (int i = 0; i < n_points; ++i) if (output[i]) inside++;

  std::cout << double(inside)/double(n_points)
            << " were inside the box.\n";

  return 0;
}