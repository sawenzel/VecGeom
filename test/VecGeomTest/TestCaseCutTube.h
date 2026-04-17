#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASECUTTUBE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASECUTTUBE_HH

#include <cmath>

#include "VecGeom/volumes/CutTube.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeCutTubeSectionInnerTestSolid()
{
  // Match the most feature-rich legacy cut-tube shape tester case: inner
  // radius, phi section, and non-parallel top and bottom cut planes.
  const vecgeom::Precision thb  = 3. * vecgeom::kPi / 4.;
  const vecgeom::Precision phib = vecgeom::kPi / 3.;
  const vecgeom::Precision tht  = vecgeom::kPi / 4.;
  const vecgeom::Precision phit = 2. * vecgeom::kPi / 3.;
  const vecgeom::Vector3D<vecgeom::Precision> nbottom(std::sin(thb) * std::cos(phib), std::sin(thb) * std::sin(phib),
                                                      std::cos(thb));
  const vecgeom::Vector3D<vecgeom::Precision> ntop(std::sin(tht) * std::cos(phit), std::sin(tht) * std::sin(phit),
                                                   std::cos(tht));
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCutTube("test-cuttube-section-inner", 3., 5., 10., 0., 2. * vecgeom::kPi / 3., nbottom, ntop));
}

} // namespace test
} // namespace vecgeom

#endif
