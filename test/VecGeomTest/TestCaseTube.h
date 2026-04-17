#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASETUBE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASETUBE_HH

#include "VecGeom/volumes/Tube.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTubeFullPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTube("test-tube-fullphi", 5., 10., 20., 0., vecgeom::kTwoPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTubeSectionTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTube("test-tube-section", 2., 6., 15., vecgeom::kPi / 7., 1.5 * vecgeom::kPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTubeThinWallLongTestSolid()
{
  // Long thin shell to stress small radial safeties against long flight
  // distances.
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTube("test-tube-thin-wall-long", 99.9, 100.0, 1000.0, 0., vecgeom::kTwoPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTubeNarrowPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTube("test-tube-narrow-phi", 5., 10., 20., vecgeom::kPi / 5., 0.01));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTubeAlmostFullPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTube("test-tube-almost-full-phi", 5., 10., 20., 0.15, vecgeom::kTwoPi - 1.e-4));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeTubeShortDiskTestSolid()
{
  // Thin z thickness stresses cap handling in a regime where distances and
  // safeties should stay close to the boundary tolerance.
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleTube("test-tube-short-disk", 2., 20., 1.e-3, 0., vecgeom::kTwoPi));
}

} // namespace test
} // namespace vecgeom

#endif
