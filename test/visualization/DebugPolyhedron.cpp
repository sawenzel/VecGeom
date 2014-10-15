#include "volumes/Polyhedron.h"
#include "utilities/ShapeDebugger.h"

using namespace vecgeom;

int main() {


  constexpr int nPlanes = 2;
  Precision zPlanes[nPlanes] = {-1, 1};
  Precision rInner[nPlanes] = {1, 1};
  Precision rOuter[nPlanes] = {2, 2};
  UnplacedPolyhedron unplaced(5, nPlanes, zPlanes, rInner, rOuter);
  LogicalVolume logical(&unplaced);
  VPlacedVolume *placed = logical.Place();
  Vector3D<Precision> bounds(3, 3, 3);

  ShapeDebugger debugger(placed);
  debugger.SetMaxMismatches(100);
  debugger.CompareDistanceToInToROOT(bounds, 4096);

  delete placed;

  return 0;
}