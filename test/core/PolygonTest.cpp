#include "volumes/Polygon.h"

using namespace vecgeom;

int main() {
  Vector2D<Precision> points[5];
  points[0].Set(1, 1);
  points[1].Set(3, 1);
  points[2].Set(4, 2);
  points[3].Set(4, 3);
  points[4].Set(1, 4);
  Polygon polygon = Polygon(points, 5);
  polygon.Print();
  return 0;
}