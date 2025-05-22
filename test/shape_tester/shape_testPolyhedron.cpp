/// \file shape_testPolyhedron.h
/// \author: Mihaela Gheata (mihaela.gheata@cern.ch)

#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include "VecGeom/volumes/Polyhedron.h"
typedef vecgeom::SimplePolyhedron Polyhedron_t;
using vecgeom::Precision;

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testPolyhedron -test <#>:\n"
                 "       0 - case 0\n"
                 "       1 - case 1\n";
  }
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 5);
  OPTION_DOUBLE(grazing, 0.);
  using namespace vecgeom;

  Polyhedron_t *solid = 0;

  auto NoInnerRadii = []() {
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-4, -2, 0, 2, 4};
    Precision rInner[nPlanes]  = {0, 0, 0, 0, 0};
    Precision rOuter[nPlanes]  = {2, 3, 2, 3, 2};
    return new Polyhedron_t("NoInnerRadii", 5, nPlanes, zPlanes, rInner, rOuter);
  };

  auto WithInnerRadii = []() {
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
    Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
    Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
    return new Polyhedron_t("WithInnerRadii", 5, nPlanes, zPlanes, rInner, rOuter);
  };

  auto WithPhiSectionConvex = []() {
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
    Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
    Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
    return new Polyhedron_t("WithPhiSectionConvex", 15 * kDegToRad, 45 * kDegToRad, 5, nPlanes, zPlanes, rInner,
                            rOuter);
  };

  auto WithPhiSectionNonConvex = []() {
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
    Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
    Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
    return new Polyhedron_t("WithPhiSectionNonConvex", 15 * kDegToRad, 340 * kDegToRad, 5, nPlanes, zPlanes, rInner,
                            rOuter);
  };

  auto ManySegments = []() {
    constexpr int nPlanes      = 17;
    Precision zPlanes[nPlanes] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    Precision rInner[nPlanes]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Precision rOuter[nPlanes]  = {2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2};
    return new Polyhedron_t("ManySegments", 6, nPlanes, zPlanes, rInner, rOuter);
  };

  auto SameZsection = []() {
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-2, -1, 1, 1, 2};
    Precision rInner[nPlanes]  = {0, 1, 0.5, 1, 0};
    Precision rOuter[nPlanes]  = {1, 2, 2, 2.5, 1};
    return new Polyhedron_t("SameZsection", 15 * kDegToRad, 340 * kDegToRad, 5, nPlanes, zPlanes, rInner, rOuter);
  };

  switch (type) {
  case 0:
    solid = NoInnerRadii();
    break;
  case 1:
    solid = WithInnerRadii();
    break;
  case 2:
    solid = WithPhiSectionConvex();
    break;
  case 3:
    solid = WithPhiSectionNonConvex();
    break;
  case 4:
    solid = ManySegments();
    break;
  case 5:
    solid = SameZsection();
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  if (!solid) return 0;
  std::cout << "Testing polyhedron: " << solid->GetName() << "\n";

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  tester.SetGrazingTolerance(grazing);
  tester.SetErrorOnZeroDoutGrazing(true);
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
  return 0;
}
