#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEEXTRUDED_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEEXTRUDED_HH

#include "VecGeom/volumes/Extruded.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeExtrudedMultiLayerTestSolid()
{
  constexpr size_t nvert = 8;
  constexpr size_t nsect = 4;
  vecgeom::XtruVertex2 vertices[nvert];
  vecgeom::XtruSection sections[nsect];

  vertices[0].x = -3.;
  vertices[0].y = -3.;
  vertices[1].x = -3.;
  vertices[1].y = 3.;
  vertices[2].x = 3.;
  vertices[2].y = 3.;
  vertices[3].x = 3.;
  vertices[3].y = -3.;
  vertices[4].x = 1.5;
  vertices[4].y = -3.;
  vertices[5].x = 1.5;
  vertices[5].y = 1.5;
  vertices[6].x = -1.5;
  vertices[6].y = 1.5;
  vertices[7].x = -1.5;
  vertices[7].y = -3.;

  sections[0].fOrigin.Set(-2., 1., -4.0);
  sections[0].fScale = 1.5;
  sections[1].fOrigin.Set(0., 0., 1.0);
  sections[1].fScale = 0.5;
  sections[2].fOrigin.Set(0., 0., 1.5);
  sections[2].fScale = 0.7;
  sections[3].fOrigin.Set(2., 2., 4.0);
  sections[3].fScale = 0.9;

  return MakeStandalonePlacedTestSolid("test-extruded-multilayer",
                                       new vecgeom::UnplacedExtruded(nvert, vertices, nsect, sections));
}

} // namespace test
} // namespace vecgeom

#endif
