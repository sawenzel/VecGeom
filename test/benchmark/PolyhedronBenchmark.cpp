#include "volumes/LogicalVolume.h"
#include "volumes/Polyhedron.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(5., 5., 5.);
  constexpr int nPlanes = 4;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  Precision rInner[nPlanes] = {1, 1, 1, 1};
  Precision rOuter[nPlanes] = {2, 2, 2, 2};
  UnplacedPolyhedron polyhedronUnplaced(4, nPlanes, zPlanes, rInner, rOuter);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume polyhedron = LogicalVolume("polyhedron", &polyhedronUnplaced);

  Transformation3D placement;
  world.PlaceDaughter("polyhedron", &polyhedron, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(8192);
  tester.SetPointCount(128);
  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  tester.RunToInBenchmark();

  return 0;
}
