#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints);
  OPTION_INT(nrep);
  OPTION_DOUBLE(dx);
  OPTION_DOUBLE(dy);
  OPTION_DOUBLE(dz);

  UnplacedBox worldUnplaced = UnplacedBox(dx*4, dy*4, dz*4);
  UnplacedBox boxUnplaced = UnplacedBox(dx, dy, dz);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume box = LogicalVolume("box", &boxUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("box", &box, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunBenchmark();

  return 0;
}
