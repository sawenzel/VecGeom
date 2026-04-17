#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASETORUS2_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASETORUS2_HH

#include "VecGeom/volumes/Torus2.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTorus2GeneralTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTorus2("test-torus2-general", 5., 10., 30., 0.25 * vecgeom::kPi, 0.75 * vecgeom::kPi));
}

} // namespace test
} // namespace vecgeom

#endif
