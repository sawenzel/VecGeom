#include "LibraryGeneric.h"
#include "Kernel.h"
#include "Box.h"

template <>
bool Box::Contains<kScalar>(Vector3D<double> const &point) const {
  return kernel::box::Contains<kScalar>(dimensions, trans_matrix, point);
}

template <>
void Box::Contains<kScalar>(SOA3D<double> const& points, bool* output) const {

  const int size = points.size();
  for (int i = 0; i < size; ++i) {
    output[i] = kernel::box::Contains<kScalar>(dimensions, trans_matrix,
                                               points[i]);
  }

}

template <>
double Box::DistanceToIn<kScalar>(Vector3D<double> const &pos,
                                  Vector3D<double> const &dir,
                                  double const step_max) const {
  return kernel::box::DistanceToIn<kScalar>(dimensions, trans_matrix, pos, dir,
                                            step_max);
}

template <>
void Box::DistanceToIn<kScalar>(SOA3D<double> const &pos,
                                SOA3D<double> const &dir,
                                double const *step_max,
                                double *distance) const {

  const int size = pos.size();
  for (int i = 0; i < size; ++i) {
    distance[i] = kernel::box::DistanceToIn<kScalar>(dimensions, trans_matrix,
                                                     pos[i], dir[i],
                                                     step_max[i]);
  }

}