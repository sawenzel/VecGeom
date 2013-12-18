#include <iostream>
#include "LibraryVc.h"
#include "LauncherVc.h"

int main(void) {
  
  const int n_points = 1<<20;

  // Populate some points
  SOA3D_Vc_Float points(n_points);
  for (int i = 0; i < points.vectors(); ++i) {
    points.Memory(0).vector(i) = VcFloat::Random() * 20. - 10.;
    points.Memory(1).vector(i) = VcFloat::Random() * 20. - 10.;
    points.Memory(2).vector(i) = VcFloat::Random() * 20. - 10.;
  }

  // Create a "box"
  const Vector3D<VcScalar> box_pos(0., 0., 0.);
  const Vector3D<VcScalar> box_dim(5., 5., 5.);

  // Output vector
  VcBool* output = (VcBool*) _mm_malloc(sizeof(VcBool)*points.vectors(),
                                        kAlignmentBoundary);

  // Launch kernel
  Stopwatch timer;
  timer.Start();
  LauncherVc::Contains(box_pos, box_dim, points, output);
  timer.Stop();
  
  std::cout << "Vc benchmark for " << n_points << " points finished in "
            << timer.Elapsed() << "s.\n";

  int inside = 0;
  for (int i = 0; i < points.vectors(); ++i) {
    const VcBool vec = output[i];
    for (int j = 0; j < VcFloat::Size; ++j) {
      if (vec[j]) inside++;
    }
  }

  std::cout << inside << " / " << n_points << " were inside the box.\n";

  return 0;
}