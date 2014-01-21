#include "LibraryGeneric.h"
#include "Kernel.h"
#include "Box.h"

bool Box::Contains(Vector3D<double> const &point) const {
  bool contains;
  kernel::box::Contains<kScalar>(dimensions, trans_matrix, point, contains);
  return contains;
}

double Box::DistanceToIn(Vector3D<double> const &pos,
                                  Vector3D<double> const &dir,
                                  double const step_max) const {
  double distance;
  kernel::box::DistanceToIn<kScalar>(dimensions, trans_matrix, pos, dir,
                                     step_max, distance);
  return distance;
}