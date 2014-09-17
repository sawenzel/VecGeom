#include "volumes/LogicalVolume.h"
#include "volumes/Polyhedron.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
  Precision zPlanes[] = {1, 3, 5, 8};
  Precision rInner[] = {1, 1, 1, 1};
  Precision rOuter[] = {3, 2, 3, 2};
  UnplacedPolyhedron polyhedronUnplaced(3, 4, zPlanes, rInner, rOuter);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume polyhedron = LogicalVolume("polyhedron", &polyhedronUnplaced);

  Transformation3D placement;
  world.PlaceDaughter("polyhedron", &polyhedron, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(999);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(1);
  tester.SetPointCount(8);
  tester.RunInsideBenchmark();
  // tester.RunToOutBenchmark();

  return 0;
}
