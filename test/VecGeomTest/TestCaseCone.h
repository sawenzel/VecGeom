#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASECONE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASECONE_HH

#include "VecGeom/volumes/Cone.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeFullPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-fullphi", 1., 6., 4., 9., 12., 0., vecgeom::kTwoPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeSectionTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-section", 0.5, 4.0, 3.0, 7.0, 10.0, vecgeom::kPi / 9., 1.3 * vecgeom::kPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeThinShellTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-thin-shell", 49.9, 50.0, 79.9, 80.0, 200.0, 0., vecgeom::kTwoPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeAlmostCylinderTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-almost-cylinder", 10.0, 20.0, 10.1, 20.1, 200.0, 0., vecgeom::kTwoPi));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeNarrowPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-narrow-phi", 0.5, 4.0, 3.0, 7.0, 10.0, 0.35, 0.01));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeAlmostFullPhiTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-almost-full-phi", 1., 6., 4., 9., 12., 0.2, vecgeom::kTwoPi - 1.e-4));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeConeClosingRingTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleCone("test-cone-closing-ring", 1238.0, 1455.22, 1455.22, 1455.22, 178.65, 0., vecgeom::kTwoPi));
}

} // namespace test
} // namespace vecgeom

#endif
