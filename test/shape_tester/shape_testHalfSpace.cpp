#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeomTest/TestCaseBoolean.h"

#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/HalfSpace.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include <iostream>
#include <memory>

namespace {

std::unique_ptr<vecgeom::VPlacedVolume> MakeHalfSpaceTestSolid(int type)
{
  switch (type) {
  case 0:
    return vecgeom::test::MakeBooleanSubtractionBoxHalfSpaceTestSolid();
  case 1: {
    auto *box       = vecgeom::test::MakeLeakedStandalonePlacedVolume("test-halfspace-oblique-box",
                                                                      new vecgeom::UnplacedBox(8., 8., 8.));
    auto *halfspace = vecgeom::test::MakeLeakedStandalonePlacedVolume(
        "test-halfspace-oblique-cutter",
        new vecgeom::UnplacedHalfSpace(vecgeom::Vector3D<vecgeom::Precision>(0.5, -0.5, 0.),
                                       vecgeom::Vector3D<vecgeom::Precision>(0., 1., 1.)));
    return vecgeom::test::MakeStandalonePlacedTestSolid(
        "test-boolean-subtraction-oblique-box-halfspace",
        new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, box, halfspace));
  }
  default:
    std::cerr << "Unknown half-space shape test type " << type << "\n";
    return nullptr;
  }
}

} // namespace

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testHalfSpace -type <#> -npoints <#>:\n"
                 "       0 - box minus half-space\n"
                 "       1 - box minus oblique half-space\n";
  }

  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 1);

  auto solid = MakeHalfSpaceTestSolid(type);
  if (!solid) return 1;

  solid->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(vecgeom::kTolerance);
  tester.SetCheckConventions(true);
  tester.SetErrorOnZeroDoutGrazing(false);

  int errCode = tester.Run(solid.get());

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  return errCode;
}
