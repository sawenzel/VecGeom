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

using namespace vecgeom;

int main() {
  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  UnplacedCone coneUnplaced = UnplacedCone(10, 20., 15, 25, 100, 0, 2.*M_PI );

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume cone  = LogicalVolume("cone", &coneUnplaced);


  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("cone", &cone, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1<<12);
  tester.RunBenchmark();

}

