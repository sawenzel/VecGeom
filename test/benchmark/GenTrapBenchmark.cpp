/*
 * GenTrapBenchmark.cpp
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 *      Modified: mihaela.gheata@cern.ch
 */
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/GenTrap.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Global.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_INT(type, 0);

  // twisted
  Precision verticesx[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};

  // no twist
  Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};

  // LAr__EMEC__OuterWheelLead02
  Precision verticesx2[8] = {0.896627857949204, -0.948748880854048, -1.63491584809173, 1.46457499100965,
                             -5.65714565287452, -7.50463942638245,  -23.6312418315791, -20.5248321431158};
  Precision verticesy2[8] = {613.710713143274, 613.710713143274, 2000.80727630046, 2000.80727630046,
                             616.052231799555, 616.052231799555, 2008.45977950464, 2008.45977950464};

  UnplacedGenTrap trapUnplaced(verticesx, verticesy, 10.);
  UnplacedGenTrap trapUnplaced1(verticesx1, verticesy1, 10);
  UnplacedGenTrap trapUnplaced2(verticesx2, verticesy2, 7.08333333333333);
  UnplacedGenTrap *trapPtr = nullptr;
  Precision dx             = 10.;
  Precision dy             = 10.;
  Precision dz             = 10.;
  switch (type) {
  case 0:
    std::cout << "________________________________________________\n"
                 " Testing twisted trapezoid for npoints = "
              << npoints << "\n________________________________________________" << std::endl;
    trapPtr = &trapUnplaced;
    break;
  case 1:
    std::cout << "________________________\n= Testing planar trapezoid for npoint s= " << npoints
              << "=\n________________________" << std::endl;
    trapPtr = &trapUnplaced1;
    break;
  case 2:
    std::cout << "________________________\n= Testing LAr__EMEC__OuterWheelLead02 trapezoid for npoint s= " << npoints
              << "=\n________________________" << std::endl;
    trapPtr = &trapUnplaced2;
    dx      = 25.;
    dy      = 2010.;
    dz      = 10.;
    break;
  default:
    std::cout << "Unknown trapezoid type" << std::endl;
    return 1;
  }
  trapPtr->Print();

  UnplacedBox worldUnplaced(dx, dy, dz);
  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume trap("gentrap", trapPtr);

  Transformation3D placement(0, 0, 0);
  world.PlaceDaughter("gentrap", &trap, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetRepetitions(nrep);
  tester.SetPoolMultiplier(1); // set this if we want to compare results
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  return tester.RunBenchmark();
}
