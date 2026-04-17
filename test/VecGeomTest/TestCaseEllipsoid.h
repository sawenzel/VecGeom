#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEELLIPSOID_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEELLIPSOID_HH

#include "VecGeom/volumes/Ellipsoid.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeEllipsoidTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleEllipsoid("test-ellipsoid", 3., 4., 5., -4.5, 3.5));
}

} // namespace test
} // namespace vecgeom

#endif
