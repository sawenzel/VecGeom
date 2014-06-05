#include "volumes/Polygon.h"

using namespace vecgeom;

int main() {
  Precision x[] = {1, 1, 1, 1, 3, 4, 4, 5, 3};
  Precision y[] = {1, 2, 2.5, 3, 4, 4, 4, 2, 1};
  Polygon polygon = Polygon(x, y, 9);
  polygon.Print();
  return 0;
}