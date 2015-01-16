#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,1024);
  OPTION_DOUBLE(dx,1.);
  OPTION_DOUBLE(dy,2.);
  OPTION_DOUBLE(dz,3.);

  UnplacedBox worldUnplaced = UnplacedBox(dx*4, dy*4, dz*4);
  UnplacedBox boxUnplaced = UnplacedBox(dx, dy, dz);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume box = LogicalVolume("box", &boxUnplaced);

  Transformation3D placement(0.1, 0, 0);
  world.PlaceDaughter("box", &box, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}
