#include "LibraryVc.h"
#include "KernelBox.h"
#include "Box.h"
#include "Tube.h"

const VcBool Impl<kVc>::kTrue = VcBool(true);
const VcBool Impl<kVc>::kFalse = VcBool(false);
const VcFloat Impl<kVc>::kZero = Vc::Zero;

void Box::Contains(SOA3D<double> const &points,
                   bool *output) const {

  const int size = points.size();
  for (int i = 0; i < size; i += kVectorSize) {
    const Vector3D<VcFloat> point(
      VcFloat(&points.x(i)),
      VcFloat(&points.y(i)),
      VcFloat(&points.z(i))
    );
    VcBool res;
    kernel::box::Contains<kVc>(parameters->dimensions, trans_matrix, point,
                               res);
    res.store(&output[i]);
  }

}

void Box::DistanceToIn(SOA3D<double> const &pos,
                       SOA3D<double> const &dir,
                       double const *steps_max,
                       double *distance) const {

  const int size = pos.size();
  for (int i = 0; i < size; i += kVectorSize) {
    const Vector3D<VcFloat> position(
      VcFloat(&pos.x(i)),
      VcFloat(&pos.y(i)),
      VcFloat(&pos.z(i))
    );
    const Vector3D<VcFloat> direction(
      VcFloat(&dir.x(i)),
      VcFloat(&dir.y(i)),
      VcFloat(&dir.z(i))
    );
    const VcFloat step_max(&steps_max[i]);
    VcFloat res;
    kernel::box::DistanceToIn<kVc>(
      parameters->dimensions, trans_matrix, position, direction, step_max, res
    );
    res.store(&distance[i]);
  }

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