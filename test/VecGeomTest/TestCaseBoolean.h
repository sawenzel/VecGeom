#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEBOOLEAN_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEBOOLEAN_HH

#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanUnionOffsetBoxesTestSolid()
{
  auto *left = MakeLeakedStandalonePlacedVolume("test-bool-union-left", new vecgeom::UnplacedBox(5., 5., 5.));
  vecgeom::Transformation3D right_transform(-2.5, 0., 3.5);
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-union-right", new vecgeom::UnplacedBox(2., 2., 10.),
                                                 &right_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-union-offset-boxes",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kUnion>(vecgeom::kUnion, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanIntersectionBoxTubeTestSolid()
{
  auto *left  = MakeLeakedStandalonePlacedVolume("test-bool-intersection-left", new vecgeom::UnplacedBox(5., 5., 5.));
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-intersection-right",
                                                 new vecgeom::GenericUnplacedTube(2., 4., 6., 0., vecgeom::kTwoPi));
  return MakeStandalonePlacedTestSolid(
      "test-boolean-intersection-box-tube",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kIntersection>(vecgeom::kIntersection, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionBoxTubeTestSolid()
{
  auto *left = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-left", new vecgeom::UnplacedBox(10., 10., 10.));
  vecgeom::Transformation3D hole_transform(-2.5, -2.5, 0.);
  auto *hole = MakeLeakedStandalonePlacedVolume(
      "test-bool-subtraction-hole", new vecgeom::GenericUnplacedTube(0., 0.9 * 10. / 4., 10., 0., vecgeom::kTwoPi),
      &hole_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-box-tube",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, left, hole));
}

} // namespace test
} // namespace vecgeom

#endif
