// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Shape test for Ellipsoid shape
/// @file test/shape_tester/shape_testEllipsoid.cpp
/// @author Evgueni Tcherniaev

#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/Ellipsoid.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGEllipsoid   = vecgeom::SimpleEllipsoid;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  OPTION_DOUBLE(dx, 3);
  OPTION_DOUBLE(dy, 4);
  OPTION_DOUBLE(dz, 5);
  OPTION_DOUBLE(zbottom, -4.5);
  OPTION_DOUBLE(ztop, 3.5);

  auto solid = new VGEllipsoid("vecgeomEllipsoid", dx, dy, dz, zbottom, ztop);
  solid->Print();
  return runTester<VPlacedVolume>(solid, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{
  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  #ifdef VECGEOM_SINGLE_PRECISION
    tester.SetSolidTolerance(1.e-4);
  #endif
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errcode;
}
