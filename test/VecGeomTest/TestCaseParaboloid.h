#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEPARABOLOID_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEPARABOLOID_HH

#include "VecGeom/volumes/Paraboloid.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeParaboloidTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleParaboloid("test-paraboloid", 6., 10., 10.));
}

} // namespace test
} // namespace vecgeom

#endif
