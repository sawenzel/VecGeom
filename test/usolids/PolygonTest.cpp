#include "volumes/Polygon.h"
#include "UReduciblePolygon.hh"

using namespace vecgeom;

int main() {
  constexpr int n = 7;
  Precision a[n] = {1, 2, 3, 4, 4, 1, -2};
  Precision b[n] = {1, 0, -1, 2, 3, 4, 2};
  // UReduciblePolygon upolygon(a, b, n);
  // upolygon.RemoveRedundantVertices(kTolerance);
  // assert(!upolygon.CrossesItself(kTolerance));
  Polygon polygon(a, b, n);
  // Assert(std::abs(polygon.SurfaceArea() - upolygon.Area()) < kTolerance);
  return 0;
}