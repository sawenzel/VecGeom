#include "volumes/Polyhedron.h"
#include "utilities/ShapeDebugger.h"

using namespace vecgeom;

int main() {

  constexpr int nPlanes = 5;
  Precision zPlanes[nPlanes] = {-2, -1, 0, 1, 2};
  Precision rInner[nPlanes] = {0.75, 0.75, 0.75, 0.75, 0.75};
  Precision rOuter[nPlanes] = {1, 1, 1, 1, 1};
  SimplePolyhedron polyhedron("Debug", 4, nPlanes, zPlanes, rInner, rOuter);
  Vector3D<Precision> bounds(2, 2, 3);

  ShapeDebugger debugger(&polyhedron);
  debugger.CompareDistanceToOutToROOT(bounds, 256);

  return 0;
}