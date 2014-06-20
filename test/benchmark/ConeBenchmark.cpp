/*
 * ConeBenchmark.cpp
 *
 *  Created on: Jun 19, 2014
 *      Author: swenzel
 */

#include "volumes/LogicalVolume.h"
#include "volumes/Cone.h"
#include "volumes/PlacedBox.h"
#include "volumes/UnplacedBox.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include <cassert>

using namespace vecgeom;

int main() {
  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  UnplacedCone coneUnplaced = UnplacedCone(0, 20., 0, 25, 100, 0, 2.*M_PI );

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume cone  = LogicalVolume("cone", &coneUnplaced);

  Transformation3D placement(0, 0, 0);
  VPlacedVolume const* plcone = world.PlaceDaughter("cone", &cone, &placement);

  // now the cone is placed; how do we get it back?
  Vector3D<Precision> globalpoint(5.,5.,5.);
  assert(plcone->Contains(globalpoint) == true);
  Vector3D<Precision> globalpoint2(-100,-100,-100);
  assert(plcone->Contains(globalpoint2) == false);
  Vector3D<Precision> globalpoint3(-12.5/1.41,-12.5/1.41,50);
  assert(plcone->Contains(globalpoint3) == true);

  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1<<12);
  tester.RunBenchmark();

}

