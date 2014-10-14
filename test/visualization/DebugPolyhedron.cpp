#include "volumes/Polyhedron.h"
#include "utilities/ShapeDebugger.h"

using namespace vecgeom;

int main() {


  constexpr int nPlanes = 4;
  Precision zPlanes[nPlanes] = {-3, -1, 1, 3};
  Precision rInner[nPlanes] = {1, 1, 1, 1};
  Precision rOuter[nPlanes] = {2, 2, 2, 2};
  UnplacedPolyhedron unplaced(4, nPlanes, zPlanes, rInner, rOuter);
  LogicalVolume logical(&unplaced);
  VPlacedVolume *placed = logical.Place();
  Vector3D<Precision> bounds(8, 8, 8);

  ShapeDebugger debugger(placed);
  debugger.CompareDistanceToInToROOT(bounds, 2048);

  delete placed;

  return 0;
}