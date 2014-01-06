#include "LibraryVc.h"
#include "Kernel.h"
#include "Box.h"

template <>
void Box::Contains<kVc>(SOA3D<double> const &points, bool *output) const {

  const int size = points.size();
  for (int i = 0; i < size; i += kVectorSize) {
    const Vector3D<VcFloat> point((points.Memory(0))[i], (points.Memory(1))[i],
                                  (points.Memory(2))[i]);
    const VcBool res = kernel::box::Contains<kVc>(trans_matrix->Translation(),
                                                  dimensions, point);
    res.store(&output[i]);
  }

}