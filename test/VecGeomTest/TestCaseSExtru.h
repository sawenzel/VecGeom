#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASESEXTRU_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASESEXTRU_HH

#include <cmath>

#include "VecGeom/volumes/SExtru.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeSExtruCircularTestSolid()
{
  constexpr size_t n = 20;
  vecgeom::Precision x[n], y[n];
  for (size_t i = 0; i < n; ++i) {
    x[i] = 4. * std::sin(i * (2. * vecgeom::kPi) / n);
    y[i] = 4. * std::cos(i * (2. * vecgeom::kPi) / n);
  }
  return std::unique_ptr<vecgeom::VPlacedVolume>(new vecgeom::SimpleSExtru("test-sextru-circular", n, x, y, -5., 10.));
}

} // namespace test
} // namespace vecgeom

#endif
