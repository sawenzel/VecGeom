#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASETRD_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASETRD_HH

#include "VecGeom/volumes/Trd.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTrdBoxlikeTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleTrd("test-trd-boxlike", 20., 20., 30., 30., 40.));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTrdIncreasingXYTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTrd("test-trd-increasing-xy", 10., 20., 20., 40., 40.));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTrdExtremeAspectTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTrd("test-trd-extreme-aspect", 0.5, 0.5, 100.0, 100.0, 500.0));
}

} // namespace test
} // namespace vecgeom

#endif
