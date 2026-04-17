#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASETESSELLATED_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASETESSELLATED_HH

#include <cmath>

#include "VecGeom/volumes/Tessellated.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTessellatedOrbTestSolid()
{
  auto *tsl               = new vecgeom::UnplacedTessellated();
  constexpr double radius = 10.;
  constexpr int ngrid     = 12;
  const double dtheta     = vecgeom::kPi / ngrid;
  const double dphi       = vecgeom::kTwoPi / ngrid;

  auto vertex = [&](int itheta, int iphi) {
    const double theta = itheta * dtheta;
    const double phi   = iphi * dphi;
    return vecgeom::Vector3D<vecgeom::Precision>(radius * std::sin(theta) * std::cos(phi),
                                                 radius * std::sin(theta) * std::sin(phi), radius * std::cos(theta));
  };

  for (int itheta = 0; itheta < ngrid; ++itheta) {
    for (int iphi = 0; iphi < ngrid; ++iphi) {
      if (itheta == 0) {
        tsl->AddTriangularFacet(vecgeom::Vector3D<vecgeom::Precision>(0., 0., radius), vertex(itheta + 1, iphi),
                                vertex(itheta + 1, iphi + 1));
      } else if (itheta == ngrid - 1) {
        tsl->AddTriangularFacet(vertex(itheta, iphi), vecgeom::Vector3D<vecgeom::Precision>(0., 0., -radius),
                                vertex(itheta, iphi + 1));
      } else {
        tsl->AddQuadrilateralFacet(vertex(itheta, iphi), vertex(itheta + 1, iphi), vertex(itheta + 1, iphi + 1),
                                   vertex(itheta, iphi + 1));
      }
    }
  }
  tsl->Close();
  return MakeStandalonePlacedTestSolid("test-tessellated-orb", tsl);
}

} // namespace test
} // namespace vecgeom

#endif
