#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEGENTRAP_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEGENTRAP_HH

#include "VecGeom/volumes/GenTrap.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeGenTrapTwistedTestSolid()
{
  vecgeom::Precision verticesx[8] = {-3., -2.5, 3., 2.5, -2., -2., 2., 2.};
  vecgeom::Precision verticesy[8] = {-2.5, 3., 2.5, -3., -2., 2., 2., -2.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleGenTrap("test-gentrap-twisted", verticesx, verticesy, 5.));
}

} // namespace test
} // namespace vecgeom

#endif
