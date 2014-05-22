#include "volumes/logical_volume.h"
#include "volumes/Tube.h"
#include "benchmarking/Benchmarker.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {
  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  UnplacedTube tubeUnplaced = UnplacedTube(10, 20., 30., 0, 3*M_PI/2);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume tube = LogicalVolume("tube", &tubeUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("tube", &tube, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1<<12);
  tester.RunBenchmark();


}
