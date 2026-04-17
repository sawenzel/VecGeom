#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEPOLYHEDRON_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEPOLYHEDRON_HH

#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakePolyhedronPhiSectionTestSolid()
{
  constexpr int nplanes               = 5;
  vecgeom::Precision zplanes[nplanes] = {-4., -1., 0., 1., 4.};
  vecgeom::Precision rinner[nplanes]  = {1., 0.75, 0.5, 0.75, 1.};
  vecgeom::Precision router[nplanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimplePolyhedron("test-polyhedron-phi-section", 15. * vecgeom::kDegToRad, 340. * vecgeom::kDegToRad,
                                    5, nplanes, zplanes, rinner, router));
}

} // namespace test
} // namespace vecgeom

#endif
