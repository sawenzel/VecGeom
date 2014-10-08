#include "volumes/LogicalVolume.h"
#include "volumes/Polyhedron.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(8., 8., 8.);
  constexpr int nPlanes = 6;
  Precision zPlanes[nPlanes] = {-4, -2, -1, 1, 2, 4};
  Precision rInner[nPlanes] = {0.5, 1, 1.5, 1.5, 1, 0.5};
  Precision rOuter[nPlanes] = {1, 2, 3, 3, 2, 1};
  UnplacedPolyhedron polyhedronUnplaced(8, nPlanes, zPlanes, rInner, rOuter);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume polyhedron = LogicalVolume("polyhedron", &polyhedronUnplaced);

  Transformation3D placement;
  world.PlaceDaughter("polyhedron", &polyhedron, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(1);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(8192);
  tester.SetPointCount(256);
  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  tester.RunToInBenchmark();

  return 0;
}
