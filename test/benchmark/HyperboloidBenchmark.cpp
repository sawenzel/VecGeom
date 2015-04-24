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
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,10);
  OPTION_DOUBLE(rmin,10.);
  OPTION_DOUBLE(rmax,15.);
  OPTION_DOUBLE(sin,30.);
  OPTION_DOUBLE(sout,30.);
  OPTION_DOUBLE(dz,50);
 

  UnplacedBox worldUnplaced = UnplacedBox(rmax*4, rmax*4, dz*4);
  UnplacedHype hypeUnplaced = UnplacedHype(rmin,sin,rmax,sout,dz);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume hype = LogicalVolume("p4r4", &hypeUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  //world.PlaceDaughter(&hype, &placement);
  world.PlaceDaughter(&hype, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  
  tester.SetTolerance(1e-6);
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
	
  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  tester.RunToInBenchmark();
  

  
  return 0;
}
