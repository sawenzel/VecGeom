#include "LibraryVc.h"
#include "Kernel.h"
#include "Box.h"

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
                            double *distance) const {

  const int size = pos.size();
  for (int i = 0; i < size; i += kVectorSize) {
    const Vector3D<VcFloat> p((pos.Memory(0))[i], (pos.Memory(1))[i],
                              (pos.Memory(2))[i]);
    const Vector3D<VcFloat> d((dir.Memory(0))[i], (dir.Memory(1))[i],
                              (dir.Memory(2))[i]);
    const VcFloat res = kernel::box::DistanceToIn<kVc>(
      dimensions, trans_matrix, p, d
    );
    res.store(&distance[i]);
  }

}