#include "volumes/Polyhedron.h"
#include "utilities/ShapeDebugger.h"

using namespace vecgeom;

int main() {

  constexpr int nPlanes = 4;
  Precision zPlanes[nPlanes] = {-3, -1, 1, 3};
  Precision rInner[nPlanes] = {0, 0, 0, 0};
  Precision rOuter[nPlanes] = {4, 4, 4, 4};
  SimplePolyhedron polyhedron("Debug", 4, nPlanes, zPlanes, rInner, rOuter);
  Vector3D<Precision> bounds(6.5, 6.5, 3.1);

  ShapeDebugger debugger(&polyhedron);
  debugger.CompareDistanceToOutToROOT(bounds, 256);

  return 0;
}