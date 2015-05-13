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
#include "ArgParser.h"
#include <cassert>

using namespace vecgeom;

int main(int argc, char * argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,1024);
  OPTION_DOUBLE(rmin1,5);
  OPTION_DOUBLE(rmax1,10);
  OPTION_DOUBLE(rmin2,7);
  OPTION_DOUBLE(rmax2,15);
  OPTION_DOUBLE(dz,10);
  OPTION_DOUBLE(sphi,0);
  OPTION_DOUBLE(dphi,kTwoPi);


  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  UnplacedCone coneUnplaced = UnplacedCone(rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi );

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume cone  = LogicalVolume("cone", &coneUnplaced);

  Transformation3D placement(5, 0, 0);
  world.PlaceDaughter("cone", &cone, &placement);

  // now the cone is placed; how do we get it back?
  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunBenchmark();

}

