#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASESPHERE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASESPHERE_HH

#include "VecGeom/volumes/Sphere.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeSphereSectionTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleSphere("test-sphere-section", 15., 20., 0., 2. * vecgeom::kPi / 3., 0., vecgeom::kPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeSphereThinShellTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleSphere("test-sphere-thin-shell", 99.9, 100.0, 0., vecgeom::kTwoPi, 0., vecgeom::kPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeSphereNarrowPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleSphere("test-sphere-narrow-phi", 5., 20., 0.4, 0.01, 0., vecgeom::kPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeSphereNarrowThetaTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleSphere("test-sphere-narrow-theta", 5., 20., 0., vecgeom::kTwoPi, 0.2, 0.01));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeSphereAlmostFullPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleSphere("test-sphere-almost-full-phi", 5., 20., 0.2,
                                                                           vecgeom::kTwoPi - 1.e-4, 0., vecgeom::kPi));
}

} // namespace test
} // namespace vecgeom

#endif
