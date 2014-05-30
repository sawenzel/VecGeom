#include "base/Vector3D.h"

#include <iostream>

namespace VECGEOM_NAMESPACE {

std::ostream& operator<<(std::ostream& os, Vector3D<Precision> const &vec) {
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
  return os;
}

} // End global namespace