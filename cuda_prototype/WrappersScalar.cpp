#include "LibraryGeneric.h"
#include "Kernel.h"
#include "Box.h"

template <>
bool Shape::Contains<Box, kScalar>(Vector3D<double> const &point) {
  return kernel::box::Contains<kScalar>(parameters->dimensions, trans_matrix,
                                        pos);
}

template <>
double Shape::DistanceToIn<Box, kScalar>(Vector3D<double> const &pos,
                                         Vector3D<double> const &dir) {
  return kernel::box::DistanceToIn<kScalar>(parameters->dimensions,
                                            trans_matrix, pos, dir);
}

template <>
void Shape::DistanceToIn<Box, kScalar>(SOA3D<double> const &pos,
                                       SOA3D<double> const &dir,
                                       double *distance) const {

  const int size = pos.size();
  for (int i = 0; i < size; ++i) {
    const Vector3D<double> p(pos.x[i], pos.y[i],
                             pos.z[i]);
    const Vector3D<double> d(dir.x[i], dir.y[i],
                             dir.z[i]);
    distance[i] =
        kernel::box::DistanceToIn<kScalar>(parameters->dimensions,
                                           trans_matrix, p, d);
  }

}