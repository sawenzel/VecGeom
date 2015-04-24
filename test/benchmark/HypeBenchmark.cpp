/// \file HypeBenchmark.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Hype.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,100);
  OPTION_INT(nrep,1);
  OPTION_DOUBLE(r,3.);

  UnplacedBox worldUnplaced = UnplacedBox(r*4, r*4, r*4);
  UnplacedHype hypeUnplaced = UnplacedHype(2,45,4,45,5);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume hype = LogicalVolume("p4r4", &hypeUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&hype, &placement);
  //world.PlaceDaughter(&hype, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  tester.RunBenchmark();

  return 0;
}
