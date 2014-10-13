#include "volumes/Polyhedron.h"
#include "utilities/ShapeDebugger.h"

using namespace vecgeom;

int main() {

  constexpr int nPlanes = 5;
  Precision zPlanes[nPlanes] = {-2, -1, 0, 1, 2};
  Precision rInner[nPlanes] = {1, 2, 1, 2, 1};
  Precision rOuter[nPlanes] = {2, 3, 2, 3, 2};
  SimplePolyhedron polyhedron("Debug", 4, nPlanes, zPlanes, rInner, rOuter);
  Vector3D<Precision> bounds(4, 4, 3);

  ShapeDebugger debugger(&polyhedron);
  debugger.CompareDistanceToInToROOT(bounds, 1024);

  return 0;
}