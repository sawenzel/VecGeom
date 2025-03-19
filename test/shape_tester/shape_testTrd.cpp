// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Shape test for Trd
/// @file test/shape_tester/shape_testTrd.cpp

#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/Trd.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTrd         = vecgeom::SimpleTrd;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

template <typename Trd_t>
Trd_t *buildATrd(int test)
{
  switch (test) {
  case 0:
    // Constant dx/dy (box)
    return new Trd_t("TrdBox", 20., 20., 30., 30., 40.);
  case 1:
    // Variable dx constant dy
    return new Trd_t("TrdConstY", 20., 30., 30., 30., 40.);
  case 2:
    // Increasing dx increasing dy
    return new Trd_t("TrdIncreasingXY", 10., 20., 20., 40., 40.);
  case 3:
    // Decreasing dx decreasing dy
    return new Trd_t("TrdDecreasingXY", 30., 10., 40., 30., 40.);
  case 4:
    // BaBar thin dx large dy
    return new Trd_t("TrdBabarThin", 0.15, 0.15, 24.707, 24.707, 7.);
  default:
    std::cout << "Unknown test case.\n";
    return 0;
  }
}

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testTrd -test <#>:\n"
                 "       0 - Constant dx/dy (box)\n"
                 "       1 - Variable dx constant dy\n"
                 "       2 - Increasing dx increasing dy\n"
                 "       3 - Decreasing dx decreasing dy\n"
                 "       4 - BaBar thin dx large dy\n";
  }

  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 0);

  auto trd = buildATrd<VGTrd>(type);
  trd->Print();
  return runTester<VPlacedVolume>(trd, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{

  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  #ifdef VECGEOM_SINGLE_PRECISION
    tester.SetSolidTolerance(1e-4);
  #endif
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errcode;
}
