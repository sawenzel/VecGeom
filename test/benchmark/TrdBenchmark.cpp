#include "volumes/LogicalVolume.h"
#include "volumes/Trd.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

double dmax(double d1, double d2) {
    if(d1 > d2) return d1;
    return d2;
}

int main(int argc, char* argv[]) {
  OPTION_INT(npoints);
  OPTION_INT(nrep);
  OPTION_DOUBLE(dx1);
  OPTION_DOUBLE(dx2);
  OPTION_DOUBLE(dy1);
  OPTION_DOUBLE(dy2);
  OPTION_DOUBLE(dz);

  UnplacedBox worldUnplaced = UnplacedBox(dmax(dx1, dx2)*4, dmax(dy1, dy2)*4, dz*4);
  UnplacedTrd trdUnplaced = UnplacedTrd(dx1, dx2, dy1, dy2, dz);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume trd = LogicalVolume("trdLogicalVolume", &trdUnplaced);
  Transformation3D placement(5., 5., 5.);
  world.PlaceDaughter("trdPlaced", &trd, &placement);


  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunBenchmark();
}
