/// \file OrbBenchmark.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Orb.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
  UnplacedOrb orbUnplaced = UnplacedOrb(3);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume orb = LogicalVolume("p4r4", &orbUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&orb, &placement);
  //world.PlaceDaughter(&orb, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPointCount(1<<12);
  tester.RunBenchmark();

  return 0;
}
