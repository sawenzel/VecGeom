#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEHYPE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEHYPE_HH

#include "VecGeom/volumes/Hype.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeHypeTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleHype("test-hype", 5., 20., vecgeom::kPi / 6., vecgeom::kPi / 3., 50.));
}

} // namespace test
} // namespace vecgeom

#endif
