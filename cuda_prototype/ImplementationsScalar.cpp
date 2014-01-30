#include "LibraryGeneric.h"
#include "KernelBox.h"
#include "Box.h"
#include "Tube.h"

void Box::Contains(SOA3D<double> const &points,
                   bool *output) const {
  // Not implemented
}

void Box::DistanceToIn(SOA3D<double> const &pos,
                       SOA3D<double> const &dir,
                       double const *steps_max,
                       double *distance) const {

  const int size = pos.size();
  for (int i = 0; i < size; ++i) {
    kernel::box::DistanceToIn<kScalar>(
      parameters->dimensions, trans_matrix, pos[i], dir[i], steps_max[i],
      distance[i]
    );
  }

}

void Tube::Contains(SOA3D<double> const &points,
                    bool *output) const {
  // Not implemented
}

void Tube::DistanceToIn(SOA3D<double> const &pos,
                        SOA3D<double> const &dir,
                        double const *steps_max,
                        double *distance) const {
  // Not implemented
}