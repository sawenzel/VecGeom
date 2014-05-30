#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
  UnplacedBox boxUnplaced = UnplacedBox(2.5, 2.5, 2.5);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume box = LogicalVolume("box", &boxUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("box", &box, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1<<13);
  tester.RunBenchmark();

  return 0;
}