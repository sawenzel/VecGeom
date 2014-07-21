#include "volumes/LogicalVolume.h"
#include "volumes/Trd.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {
  UnplacedBox worldUnplaced = UnplacedBox(1000., 1000., 1000.);
  UnplacedTrd trdUnplaced = UnplacedTrd(5., 10., 9., 4., 30.);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume trd = LogicalVolume("trdLogicalVolume", &trdUnplaced);
  Transformation3D placement(5., 5., 5.);
  world.PlaceDaughter("trdPlaced", &trd, &placement);


  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1<<10);
  tester.RunToInBenchmark();
}
