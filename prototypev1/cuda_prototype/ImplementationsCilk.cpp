#include "LibraryCilk.h"
#include "KernelBox.h"
#include "Box.h"
#include "Tube.h"

void Box::DistanceToIn(SOA3D<double> const &pos,
                       SOA3D<double> const &dir,
                       double const *steps_max,
                       double *distance) const {

  const int size = pos.size();
  constexpr int vec_size = CilkFloat::VecSize();
  const int i_max = size - (size % vec_size);
  for (int i = 0; i < i_max; i += vec_size) {
    const Vector3D<CilkFloat> pos_cilk(
      CilkFloat(&pos.x(i)),
      CilkFloat(&pos.y(i)),
      CilkFloat(&pos.z(i))
    );
    const Vector3D<CilkFloat> dir_cilk(
      CilkFloat(&dir.x(i)),
      CilkFloat(&dir.y(i)),
      CilkFloat(&dir.z(i))
    );
    const CilkFloat step_max(steps_max);
    CilkFloat output;
    kernel::box::DistanceToIn<kCilk>(
      parameters->dimensions, trans_matrix, pos_cilk, dir_cilk, step_max, output
    );
    output.Store(&distance[i]);
  }
  for (int i = i_max; i < size; ++i) {
    kernel::box::DistanceToIn<kScalar>(
      parameters->dimensions, trans_matrix, pos[i], dir[i], steps_max[i],
      distance[i]
    );
  }

}

void Box::Contains(SOA3D<double> const &points,
                   bool *output) const {
  // NYI
}

void Tube::Contains(SOA3D<double> const &points,
                    bool *output) const {
  // NYI
}

void Tube::DistanceToIn(SOA3D<double> const &pos,
                        SOA3D<double> const &dir,
                        double const *steps_max,
                        double *distance) const {
  // NYI
}