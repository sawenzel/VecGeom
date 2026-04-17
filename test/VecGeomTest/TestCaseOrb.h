#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEORB_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEORB_HH

#include "VecGeom/volumes/Orb.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeOrbTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleOrb("test-orb", 8.));
}

} // namespace test
} // namespace vecgeom

#endif
