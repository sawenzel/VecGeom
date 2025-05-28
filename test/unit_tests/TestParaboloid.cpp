// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for Paralleliped.
/// @file test/unit_tests/TestParaboloid.cpp
/// @author Raman Sehgal

#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Paraboloid.h"
#include "ApproxEqual.h"

#include <cmath>
#include <iomanip>

#define PI 3.14159265358979323846

template <class Paraboloid_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestParaboloid()
{
  Paraboloid_t p1("testPara", 6., 9., 10.);
  // std::cout<< p1.GetK1() << std::endl;
  // VECGEOM_ASSERT(false);

  std::cout << std::setprecision(15);
  Vec_t norm(0., 0., 0.);
  double Dist = 0;

  Vec_t pzero(0., 0., 0.), pbigx(100, 0., 0.), pbigmx(-100., 0., 0.), pbigy(0., 100., 0.), pbigmy(0., -100., 0.),
      pbigz(0., 0., 100.), pbigmz(0., 0., -100);
  Vec_t dirx(1., 0., 0.), dirmx(-1., 0., 0.), diry(0., 1., 0.), dirmy(0., -1., 0.), dirz(0., 0., 1), dirmz(0., 0., -1.);

  Vec_t zSurfPt(0., 0., 10.), mzSurfPt(0., 0., -10.);

  // Inside
  VECGEOM_ASSERT(p1.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(p1.Inside(pbigx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(p1.Inside(pbigmx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(p1.Inside(pbigy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(p1.Inside(pbigmy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(p1.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(p1.Inside(pbigmz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(p1.Inside(zSurfPt) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(p1.Inside(mzSurfPt) == vecgeom::EInside::kSurface);

  Vec_t pInsideZ(0., 0., 10. - 2 * vecgeom::kTolerance);
  VECGEOM_ASSERT(p1.Inside(pInsideZ) == vecgeom::EInside::kInside);
  Vec_t pOutsideZ(0., 0., 10. + 2 * vecgeom::kTolerance);
  VECGEOM_ASSERT(p1.Inside(pOutsideZ) == vecgeom::EInside::kOutside);
  Vec_t pWithinZTolerance(0., 0., 10. + 0.25 * vecgeom::kTolerance);
  VECGEOM_ASSERT(p1.Inside(pWithinZTolerance) == vecgeom::EInside::kSurface);

  Dist = p1.DistanceToOut(pzero, dirz);
  VECGEOM_ASSERT(Dist == 10.);
  Dist = p1.DistanceToOut(pzero, dirmz);
  VECGEOM_ASSERT(Dist == 10.);
  Dist = p1.DistanceToOut(zSurfPt, dirz);
  VECGEOM_ASSERT(Dist == 0.);
  Dist = p1.DistanceToOut(mzSurfPt, dirmz);
  VECGEOM_ASSERT(Dist == 0.);
  Dist = p1.DistanceToIn(pbigz, dirmz);
  VECGEOM_ASSERT(Dist == 90.);
  Dist = p1.DistanceToIn(pbigmz, dirz);
  VECGEOM_ASSERT(Dist == 90.);
  Dist = p1.DistanceToIn(zSurfPt, dirmz);
  VECGEOM_ASSERT(Dist == 0.);
  Dist = p1.DistanceToIn(mzSurfPt, dirz);
  VECGEOM_ASSERT(Dist == 0.);

  Paraboloid_t p2("testPara", 0., 8., 10.);
  Dist = p2.DistanceToOut(Vec_t(0., 0., -10), dirmz);
  VECGEOM_ASSERT(Dist == 0.);
  Dist = p2.DistanceToOut(Vec_t(0., 0., -10), dirz);
  VECGEOM_ASSERT(Dist == 20.);
  Dist = p2.DistanceToIn(Vec_t(0., 0., -10), dirz);
  VECGEOM_ASSERT(Dist == 0.);

  Vec_t tmpDir = Vec_t(1., 2., 2.).Unit();
  Dist         = p2.DistanceToOut(pzero, tmpDir);
  Vec_t tmpPt  = pzero + Dist * tmpDir;
  Dist         = p2.DistanceToOut(tmpPt, tmpDir);
  VECGEOM_ASSERT(Dist == 0.);
  Dist = p2.DistanceToIn(tmpPt, -tmpDir);
  VECGEOM_ASSERT(Dist == 0.);

  Vec_t normal(0., 0., 0.);
  Vec_t pTopZ(0., 0., 10), pBottomZ(0., 0., -10.);
  bool valid = p1.Normal(pTopZ, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, 1.)));
  valid = p1.Normal(pBottomZ, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, -1.)));
  pTopZ.Set(3., 4, 10.);
  valid = p1.Normal(pTopZ, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, 1.)));
  pBottomZ.Set(3., 4, -10.);
  valid = p1.Normal(pBottomZ, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, -1.)));
  pTopZ.Set(9., 0., 10.);
  valid = p1.Normal(pTopZ, normal);
  VECGEOM_ASSERT(valid);

  // Testing noraml for outside point,
  // Logic to calculate normal for this case is not yet written.
  // This test is just to check the "valid", which should be false
  pTopZ.Set(9., 0., 5.);
  valid = p1.Normal(pTopZ, normal);
  VECGEOM_ASSERT(!valid);

  Vec_t pOutZ(0., 0., 11.);
  valid = p1.Normal(pOutZ, normal);
  VECGEOM_ASSERT(!valid && ApproxEqual(normal, Vec_t(0, 0, 1.)));

  pOutZ.Set(0., 0., -11.);
  valid = p1.Normal(pOutZ, normal);
  VECGEOM_ASSERT(!valid && ApproxEqual(normal, Vec_t(0, 0, -1.)));

  // Check Extent and cached BBox
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  p1.Extent(minExtent, maxExtent);
  p1.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));

  return true;
}

int main(int argc, char *argv[])
{
  VECGEOM_ASSERT(TestParaboloid<vecgeom::SimpleParaboloid>());
  std::cout << "VecGeomParaboloid passed\n";

  return 0;
}
