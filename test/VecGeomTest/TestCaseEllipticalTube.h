#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEELLIPTICALTUBE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEELLIPTICALTUBE_HH

#include "VecGeom/volumes/EllipticalTube.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeEllipticalTubeTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleEllipticalTube("test-elliptical-tube", 5., 4., 1.));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeEllipticalTubeThinTestSolid()
{
  // Flatten the elliptical tube into a thin slab to stress z-cap handling and
  // very small SafetyToIn/SafetyToOut values.
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleEllipticalTube("test-elliptical-tube-thin", 20., 12., 0.05));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeEllipticalTubeLongTestSolid()
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleEllipticalTube("test-elliptical-tube-long", 12., 4., 800.));
}

} // namespace test
} // namespace vecgeom

#endif
