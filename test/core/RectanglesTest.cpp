#include "volumes/Rectangles.h"

using namespace vecgeom;

int main() {
  typedef Vector3D<Precision> Vec_t;
  Rectangles rectangles(6);
  rectangles.Set(0, Vec_t(1, 1, 1), Vec_t(1, -1, 1), Vec_t(1, -1, -1));
  rectangles.Set(1, Vec_t(-1, 1, 1), Vec_t(-1, -1, 1), Vec_t(-1, -1, -1));
  rectangles.Set(2, Vec_t(1, 1, 1), Vec_t(-1, 1, 1), Vec_t(-1, 1, -1));
  rectangles.Set(3, Vec_t(1, -1, 1), Vec_t(-1, -1, 1), Vec_t(-1, -1, -1));
  rectangles.Set(4, Vec_t(1, 1, 1), Vec_t(-1, 1, 1), Vec_t(-1, -1, 1));
  rectangles.Set(5, Vec_t(1, 1, -1), Vec_t(-1, 1, -1), Vec_t(-1, -1, -1));
  Vec_t point(0, 0.9, 0.7);
  Vec_t direction(-1, -1.2, -1);
  direction.Normalize();
  std::cout << rectangles << point << "--" << direction << ": "
            << rectangles.Distance<true>(point, direction) << "\n";
  return 0;
}