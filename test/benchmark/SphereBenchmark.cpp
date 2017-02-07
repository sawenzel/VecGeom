/// \file SphereBenchmark.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Sphere.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,10);
  OPTION_DOUBLE(rmin,10.);
  OPTION_DOUBLE(rmax,15.);
  OPTION_DOUBLE(sphi,0.);
  OPTION_DOUBLE(dphi,2*kPi/3);
  OPTION_DOUBLE(stheta,kPi/3);
  OPTION_DOUBLE(dtheta,kPi/6); 
  //OPTION_DOUBLE(stheta,0);
  //OPTION_DOUBLE(dtheta,kPi); 

  UnplacedBox worldUnplaced = UnplacedBox(rmax*4, rmax*4, rmax*4);
  UnplacedSphere sphereUnplaced = UnplacedSphere(rmin,rmax,sphi,dphi,stheta,dtheta);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume sphere = LogicalVolume("p4r4", &sphereUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  //world.PlaceDaughter(&sphere, &placement);
  world.PlaceDaughter(&sphere, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);


  Benchmarker tester(GeoManager::Instance().GetWorld());
    tester.SetTolerance(1e-6);
  tester.SetVerbosity(3);
  //tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
	
//  tester.RunInsideBenchmark();
//  tester.RunToOutBenchmark();
//  tester.RunToInBenchmark();
  tester.RunBenchmark();
  return 0;
}
