#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/GenTrap.h"
typedef vecgeom::SimpleGenTrap GenTrap_t;

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testGenTrap -test <#>:\n"
                 "       0 - twisted\n"
                 "       1 - planar\n"
                 "       2 - one face triangle\n"
                 "       3 - one face line\n"
                 "       4 - one face point\n"
                 "       5 - one face line, other triangle\n"
                 "       6 - degenerated planar\n";
  }
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 0);

  using namespace vecgeom;
  // 4 different vertices, twisted
  Precision verticesx0[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy0[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};
  // 4 different vertices, planar
  Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};
  // 3 different vertices
  Precision verticesx2[8] = {-3, -3, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy2[8] = {-2.5, -2.5, 2.5, -3, -2, 2, 2, -2};
  // 2 different vertices
  Precision verticesx3[8] = {-3, -3, 2.5, 2.5, -2, -2, 2, 2};
  Precision verticesy3[8] = {-2.5, -2.5, -3, -3, -2, 2, 2, -2};
  // 1 vertex (pyramid)
  Precision verticesx4[8] = {-3, -3, -3, -3, -2, -2, 2, 2};
  Precision verticesy4[8] = {-2.5, -2.5, -2.5, -2.5, -2, 2, 2, -2};
  // 2 vertex bottom, 3 vertices top
  Precision verticesx5[8] = {-3, -3, 2.5, 2.5, -2, -2, 2, 2};
  Precision verticesy5[8] = {-2.5, -2.5, -3, -3, -2, -2, 2, -2};
  //
  Precision verticesx6[8] = {-0.507492, -0.507508, 1.522492, -0.507492, -0.507492, -0.507508, 1.522492, -0.507492};
  Precision verticesy6[8] = {-3.634000, 3.63400, 3.634000, -3.634000, -3.634000, 3.634000, 3.634000, -3.634000};
  GenTrap_t *solid        = 0;
  switch (type) {
  case 0:
    // 4 different vertices, twisted
    std::cout << "Testing twisted trapezoid\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx0, verticesy0, 5);
    break;
  case 1:
    // 4 different vertices, planar
    std::cout << "Testing planar trapezoid\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx1, verticesy1, 5);
    break;
  case 2:
    // 3 different vertices
    std::cout << "Testing trapezoid with one face triangle\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx2, verticesy2, 5);
    break;
  case 3:
    // 2 different vertices
    std::cout << "Testing trapezoid with one face line degenerated\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx3, verticesy3, 5);
    break;
  case 4:
    // 1 vertex (pyramid)
    std::cout << "Testing trapezoid with one face point degenerated (pyramid)\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx4, verticesy4, 5);
    break;
  case 5:
    // 2 vertex bottom, 3 vertices top
    std::cout << "Testing trapezoid with line on one face and triangle on other\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx5, verticesy5, 5);
    break;
  case 6:
    // 3 vertexes top, 3 vertexes bottom
    std::cout << "Testing degenerated planar trapezoid\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx6, verticesy6, 5);
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  solid->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);

  #ifndef VECGEOM_SINGLE_PRECISION
    tester.SetSolidTolerance(1.e-7);
  #endif
  
  tester.SetTestBoundaryErrors(true);
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
  return 0;
}
