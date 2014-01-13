#include "LibraryVc.h"
#include "Kernel.h"
#include "Box.h"

const VcBool ImplTraits<kVc>::kTrue = VcBool(true);
const VcBool ImplTraits<kVc>::kFalse = VcBool(false);
const VcFloat ImplTraits<kVc>::kZero = Vc::Zero;

template <>
void Box::Contains<kVc>(SOA3D<double> const &points,
                        bool *output) const {

  const int size = points.size();
  for (int i = 0; i < size; i += kVectorSize) {
    const Vector3D<VcFloat> point((points.Memory(0))[i], (points.Memory(1))[i],
                                  (points.Memory(2))[i]);
    const VcBool res = kernel::box::Contains<kVc>(
      dimensions, trans_matrix, point
    );
    res.store(&output[i]);
  }

}

template <>
void Box::DistanceToIn<kVc>(SOA3D<double> const &pos,
                            SOA3D<double> const &dir,
                            double const *steps_max,
                            double *distance) const {

  const int size = pos.size();
  for (int i = 0; i < size; i += kVectorSize) {
    const Vector3D<VcFloat> p(pos.x(i), pos.y(i), pos.z(i));
    const Vector3D<VcFloat> d(dir.x(i), dir.y(i), dir.z(i));
    const VcFloat step_max(steps_max[i]);
    const VcFloat res = kernel::box::DistanceToIn<kVc>(
      dimensions, trans_matrix, p, d, step_max
    );
    res.store(&distance[i]);
  }

}