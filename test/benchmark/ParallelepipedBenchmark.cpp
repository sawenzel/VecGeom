#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Parallelepiped.h"
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
  OPTION_DOUBLE(alpha);
  OPTION_DOUBLE(theta);
  OPTION_DOUBLE(phi);

  UnplacedBox worldUnplaced = UnplacedBox(dx*4, dy*4, dz*4);
  UnplacedParallelepiped paraUnplaced(dx, dy, dz, alpha, theta, phi);
  // UnplacedParallelepiped paraUnplaced(3., 3., 3., 14.9, 39, 3.22);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume para = LogicalVolume("p4r4", &paraUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&para, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunBenchmark();

  return 0;
}
