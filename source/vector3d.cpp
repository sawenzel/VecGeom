#include "base/vector3d.h"

#include <iostream>

namespace VECGEOM_NAMESPACE {

template <typename Type>
std::ostream& operator<<(std::ostream& os, Vector3D<Type> const &vec) {
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
  return os;
}

} // End global namespace