#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEPARALLELEPIPED_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEPARALLELEPIPED_HH

#include "VecGeom/volumes/Parallelepiped.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeParallelepipedGeneralTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleParallelepiped("test-parallelepiped-general", 10., 7., 15., 30., 30., 45.));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeParallelepipedHighShearTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleParallelepiped("test-parallelepiped-high-shear", 12., 6., 20., 55., 60., 75.));
}

} // namespace test
} // namespace vecgeom

#endif
