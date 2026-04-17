#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEMULTIUNION_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEMULTIUNION_HH

#include "VecGeom/volumes/MultiUnion.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeMultiUnionBoxesTestSolid()
{
  auto *multi = new vecgeom::UnplacedMultiUnion();

  multi->AddNode(new vecgeom::UnplacedBox(3., 3., 3.), vecgeom::Transformation3D(-4., 0., 0.));
  multi->AddNode(new vecgeom::UnplacedBox(3., 3., 3.), vecgeom::Transformation3D(4., 0., 0.));
  multi->AddNode(new vecgeom::UnplacedBox(2., 5., 2.), vecgeom::Transformation3D(0., 0., 0.));
  multi->AddNode(new vecgeom::UnplacedBox(1.5, 1.5, 6.), vecgeom::Transformation3D(0., 3.5, 0.));
  multi->Close();
  return MakeStandalonePlacedTestSolid("test-multiunion-boxes", multi);
}

} // namespace test
} // namespace vecgeom

#endif
