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
  OPTION_INT(npoints);
  OPTION_INT(nrep);
  OPTION_DOUBLE(rmin);
  OPTION_DOUBLE(rmax);
  OPTION_DOUBLE(sphi);
  OPTION_DOUBLE(dphi);
  OPTION_DOUBLE(stheta);
  OPTION_DOUBLE(dtheta);

  UnplacedBox worldUnplaced = UnplacedBox(rmax*4, rmax*4, rmax*4);
  UnplacedSphere sphereUnplaced = UnplacedSphere(rmin,rmax,sphi,dphi,stheta,dtheta);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume sphere = LogicalVolume("p4r4", &sphereUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&sphere, &placement);
  //world.PlaceDaughter(&sphere, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  tester.RunBenchmark();

  return 0;
}
