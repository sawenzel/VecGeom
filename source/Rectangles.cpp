#include "volumes/Rectangles.h"

namespace VECGEOM_NAMESPACE {

Rectangles::Rectangles(int size)
    : fPlanes(size), fSides(size), fCorners{size, size} {}

Rectangles::~Rectangles() {}

std::ostream& operator<<(std::ostream &os, Rectangles const &rhs) {
  for (int i = 0, iMax = rhs.size(); i < iMax; ++i) {
    Vector3D<Precision> normal = rhs.GetNormal(i);
    os << "{(" << normal[0] << ", " << normal[1] << ", " << normal[2] << ", "
       << rhs.GetDistance(i) << ") at " << rhs.GetCenter(i) << ", corners in "
       << rhs.GetCorner(0, i) << " and " << rhs.GetCorner(1, i) << ", side "
       << rhs.GetSide(i) << "}\n";
  }
  return os;
}

}