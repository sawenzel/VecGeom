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
  OPTION_INT(nrep,1024);
  OPTION_DOUBLE(rmin,1.);
  OPTION_DOUBLE(rmax,3.);
  OPTION_DOUBLE(sphi,0.);
  OPTION_DOUBLE(dphi,kPi);
  OPTION_DOUBLE(stheta,0);
  OPTION_DOUBLE(dtheta,kPi);

  UnplacedBox worldUnplaced = UnplacedBox(rmax*4, rmax*4, rmax*4);
  UnplacedSphere sphereUnplaced = UnplacedSphere(rmin,rmax,sphi,dphi,stheta,dtheta);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume sphere = LogicalVolume("p4r4", &sphereUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&sphere, &placement);
  //world.PlaceDaughter(&sphere, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  //tester.RunToInBenchmark();

  return 0;
}
