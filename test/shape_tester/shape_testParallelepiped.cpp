// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Shape test for Paralleliped.
/// @file test/shape_tester/shape_testParallelepiped.cpp
/// @author: Ceated by Mihaela Gheata

#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Parallelepiped.h"
typedef vecgeom::SimpleParallelepiped Para_t;

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testParallelepiped -test <#>:\n"
                 "       0 - alpha, theta, phi\n"
                 "       1 - alpha=0 theta=0 phi=0 (box)\n";
  }

  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 0);
  OPTION_DOUBLE(grazing, 0.);

  using namespace vecgeom;

  Para_t *solid   = 0;
  Precision dx    = 10.;
  Precision dy    = 7.;
  Precision dz    = 15.;
  Precision alpha = 30.;
  Precision theta = 30.;
  Precision phi   = 45.;

  switch (type) {
  case 0:
    // Non-zero alpha, theta, phi
    std::cout << "Testing general parallelepiped\n";
    solid = new Para_t("test_VecGeomPara", dx, dy, dz, alpha, theta, phi);
    break;
  case 1:
    // Parallelepiped degenerated to box
    std::cout << "Testing box-like parallelepiped\n";
    solid = new Para_t("test_VecGeomPara", dx, dy, dz, 0, 0, 0);
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  tester.SetGrazingTolerance(grazing);
  tester.SetErrorOnZeroDoutGrazing(true);
#ifdef VECGEOM_SINGLE_PRECISION
  tester.SetSolidTolerance(1.e-4);
#endif
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
  return 0;
}
