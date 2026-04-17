#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEBOX_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEBOX_HH

#include "VecGeom/volumes/Box.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBoxTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleBox("test-box", 10., 15., 20.));
}

} // namespace test
} // namespace vecgeom

#endif
