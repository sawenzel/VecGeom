// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Shape test for Tetrahedron
/// @file test/shape_tester/shape_testTet.cpp
/// @author Raman Sehgal

#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Tet.h"
#include "VecGeom/base/Vector3D.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTet         = vecgeom::SimpleTet;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat, double grazing_tolerance);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_DOUBLE(grazing, 0.);

  vecgeom::Vector3D<double> p0(0., 0., 2.), p1(0., 0., 0.), p2(2., 0., 0.), p3(0., 2., 0.);

  auto tet = new VGTet("vecgeomTet", p0, p1, p2, p3);
  tet->Print();
  return runTester<VPlacedVolume>(tet, npoints, debug, stat, grazing);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat, double grazing_tolerance)
{
  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  tester.SetSolidTolerance(0.5 * vecgeom::kTolerance);
  tester.SetGrazingTolerance(grazing_tolerance);
  tester.SetErrorOnZeroDoutGrazing(true);
  int errCode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errCode;
}
