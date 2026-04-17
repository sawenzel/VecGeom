#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEELLIPTICALCONE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEELLIPTICALCONE_HH

#include "VecGeom/volumes/EllipticalCone.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeEllipticalConeTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleEllipticalCone("test-elliptical-cone", 0.5, 0.4, 10., 5.));
}

} // namespace test
} // namespace vecgeom

#endif
