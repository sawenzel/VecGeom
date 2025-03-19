// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Shape test for Paraboloid
/// @file test/shape_tester/shape_testParaboloid.cpp
/// @author Raman Sehgal

#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Paraboloid.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGParaboloid  = vecgeom::SimpleParaboloid;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  OPTION_DOUBLE(rmin, 6.);
  OPTION_DOUBLE(rmax, 10.);
  OPTION_DOUBLE(dz, 10.);

  auto paraboloid = new VGParaboloid("vecgeomParaboloid", rmin, rmax, dz);
  paraboloid->Print();
  return runTester<VPlacedVolume>(paraboloid, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{
  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(true);
  #ifdef VECGEOM_SINGLE_PRECISION
     tester.SetSolidTolerance(1.e-4);
  #endif
  int errCode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errCode;
}
