#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASETET_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASETET_HH

#include "VecGeom/volumes/Tet.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTetTestSolid()
{
  // Keep the same vertex ordering as the legacy shape tester so helper-based
  // failures can be compared directly against the existing executable.
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTet("test-tet", vecgeom::Vector3D<double>(0., 0., 2.), vecgeom::Vector3D<double>(0., 0., 0.),
                             vecgeom::Vector3D<double>(2., 0., 0.), vecgeom::Vector3D<double>(0., 2., 0.)));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTetSliverLikeTestSolid()
{
  // Use a very small altitude over a wide base so navigation still sees a
  // valid tetrahedron, but with a much harsher aspect ratio than the baseline
  // helper case.
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleTet(
      "test-tet-sliver-like", vecgeom::Vector3D<double>(0., 0., 0.05), vecgeom::Vector3D<double>(0., 0., 0.),
      vecgeom::Vector3D<double>(100., 0., 0.), vecgeom::Vector3D<double>(0.1, 100., 0.)));
}

} // namespace test
} // namespace vecgeom

#endif
