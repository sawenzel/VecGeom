#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASETRAPEZOID_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASETRAPEZOID_HH

#include "VecGeom/volumes/Trapezoid.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTrapezoidFromCornersTestSolid()
{
  // Reuse the slanted-corner legacy shape tester case to cover a less regular
  // topology than the already enabled parallelepiped family.
  vecgeom::TrapCorners xyz;
  const vecgeom::Precision xoffset = 9.;
  const vecgeom::Precision yoffset = -6.;
  xyz[0]                           = vecgeom::Vector3D<vecgeom::Precision>(-2. + xoffset, -5. + yoffset, -15.);
  xyz[1]                           = vecgeom::Vector3D<vecgeom::Precision>(2. + xoffset, -5. + yoffset, -15.);
  xyz[2]                           = vecgeom::Vector3D<vecgeom::Precision>(-3. + xoffset, 5. + yoffset, -15.);
  xyz[3]                           = vecgeom::Vector3D<vecgeom::Precision>(3. + xoffset, 5. + yoffset, -15.);
  xyz[4]                           = vecgeom::Vector3D<vecgeom::Precision>(-4. - xoffset, -10. - yoffset, 15.);
  xyz[5]                           = vecgeom::Vector3D<vecgeom::Precision>(4. - xoffset, -10. - yoffset, 15.);
  xyz[6]                           = vecgeom::Vector3D<vecgeom::Precision>(-6. - xoffset, 10. - yoffset, 15.);
  xyz[7]                           = vecgeom::Vector3D<vecgeom::Precision>(6. - xoffset, 10. - yoffset, 15.);
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleTrapezoid("test-trapezoid-corners", xyz));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTrapezoidExtremeSkewTestSolid()
{
  // Reuse the strongly sheared but valid trap shape exercised in the unit
  // tests so the stress case stays aggressive without tripping planarity
  // warnings in the constructor.
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleTrapezoid(
      "test-trapezoid-extreme-skew", 50., 0., 0., 50., 50., 50., vecgeom::kPi / 4., 50., 50., 50., vecgeom::kPi / 4.));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTrapezoidNearParallelepipedTestSolid()
{
  // Stay very close to a parallelepiped with small but non-zero tilts.
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleTrapezoid(
      "test-trapezoid-near-para", 20., 0.02, 0.04, 8., 10., 10., 0.01, 8., 10., 10., 0.01));
}

} // namespace test
} // namespace vecgeom

#endif
