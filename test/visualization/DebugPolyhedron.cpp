#include "volumes/Polyhedron.h"
#include "utilities/ShapeDebugger.h"

#include <memory>

using namespace vecgeom;

int main() {


  constexpr int nPlanes = 4;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  Precision rInner[nPlanes] = {1, 1, 1, 1};
  Precision rOuter[nPlanes] = {2, 2, 2, 2};
  UnplacedPolyhedron unplaced(0, 270, 5, nPlanes, zPlanes, rInner, rOuter);
  LogicalVolume logical(&unplaced);
  Transformation3D translation(0, 0, 0);
  std::unique_ptr<VPlacedVolume> placed(logical.Place(&translation));
  Vector3D<Precision> bounds(3, 3, 3);

  ShapeDebugger debugger(placed.get());
  debugger.SetMaxMismatches(10);
  debugger.ShowCorrectResults(false);
  debugger.CompareDistanceToOutToROOT(bounds, 50000);

  return 0;
}