//
// File:    TestGenTrap.cpp
// Purpose: Unit tests for the generic trapezoid
//

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "ApproxEqual.h"

#include "VecGeom/volumes/GenTrap.h"

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"
#include "VecGeom/base/Assert.h"

bool testvecgeom = false;

bool TestGenTrap()
{
  using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
  using namespace vecgeom;
  // Planar
  Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};
  // Twisted
  Precision verticesx2[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy2[8] = {-3, 3, 3, -3, -1.9, 2, 2, -2};

  SimpleGenTrap trap1("planar_trap", verticesx1, verticesy1, 5);
  SimpleGenTrap trap2("twisted_trap", verticesx2, verticesy2, 5);

  // Some particular points
  Vec_t pzero(0, 0, 0);
  Vec_t ponxm(-2.5, 0, 0), ponxp(2.5, 0, 0), ponym(0, -2.5, 0), ponyp(0, 2.5, 0), ponzm(0, 0, -5), ponzp(0, 0, 5);
  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100), pbig(100, 100, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Precision vol, volCheck1, volCheck2;
  Precision surf, surfCheck;

  // Check cubic volume
  vol = trap1.Capacity();
  // (a*a+a*b+b*b)*h/3
  volCheck1 = (1. / 3) * (6 * 6 + 4 * 6 + 4 * 4) * 10;
  // std::cout << "volume1= " << vol << "   volCheck= " << volCheck1 << std::endl;
  VECGEOM_VALIDATE(ApproxEqual<Precision>(vol, volCheck1), << "vol != volCheck1");

  vol       = trap2.Capacity();
  volCheck2 = (1. / 3) * (6 * 6 + 3.9 * 6 + 3.9 * 3.9) * 10;
  // std::cout << "volume2= " << vol << "   should be in range: ( " << volCheck1 << ", " << volCheck2 << ")" <<
  // std::endl;
  VECGEOM_VALIDATE(vol < volCheck1 && vol > volCheck2, << "vol not between (volCheck2, volCheck1)");

  // Check surface area

  surf      = trap1.SurfaceArea();
  surfCheck = 6 * 6 + 4 * 4 + 4 * 0.5 * (6 + 4) * std::sqrt(10 * 10 + 1 * 1);
  // std::cout << "surface1= " << surf << "   surfCheck= " << surfCheck << std::endl;
  VECGEOM_VALIDATE(ApproxEqual<Precision>(surf, surfCheck), << "surf != surfcheck");

  surf = trap2.SurfaceArea();
  // std::cout << "surface2= " << surf << std::endl;
  VECGEOM_ASSERT(surf < surfCheck);

  // Check Inside

  VECGEOM_ASSERT(trap1.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(trap1.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(trap1.Inside(ponxm) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trap1.Inside(ponym) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trap1.Inside(ponzm) == vecgeom::EInside::kSurface);

  // Check Surface Normal
  bool valid;
  Vec_t normal;
  Precision phi = std::atan2(1., 10.);

  valid = trap1.Normal(ponxm, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-std::cos(phi), 0., std::sin(phi))));
  valid = trap1.Normal(ponxp, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(std::cos(phi), 0., std::sin(phi))));
  valid = trap1.Normal(ponym, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, -std::cos(phi), std::sin(phi))));
  valid = trap1.Normal(ponyp, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., std::cos(phi), std::sin(phi))));
  valid = trap1.Normal(ponzp, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., 0., 1.)));
  valid = trap1.Normal(ponzm, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., 0., -1.)));
  //    valid=trap1.Normal(pzero,normal);
  VECGEOM_ASSERT(valid == true);

  // SafetyToOut(P)
  Precision Dist, Distref;
  Dist = trap2.SafetyToOut(pzero);
  VECGEOM_ASSERT(Dist <= 2.5);
  Dist = trap2.SafetyToOut(Vec_t(0., 0., -3));
  VECGEOM_ASSERT(Dist <= 2);
  Dist = trap1.SafetyToOut(ponxm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));
  Dist = trap1.SafetyToOut(ponzp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  // DistanceToOut(P,V)
  Vec_t direction;
  Dist = trap1.DistanceToOut(pzero, Vec_t(-1., 0., 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 2.5));
  Dist = trap1.DistanceToOut(pzero, Vec_t(0., 0., 1.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 5.));
  Dist = trap1.DistanceToOut(ponxm, Vec_t(1., 0., 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 5.));
  Dist = trap1.DistanceToOut(ponyp, Vec_t(0., -1., 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 5.));
  Dist = trap1.DistanceToOut(pbig, Vec_t(0., -1., 0.));
  VECGEOM_ASSERT(Dist < 0.);
  for (int i = 0; i < 8; i++) {
    // Shoot to every vertex of the twisted trapezoid
    direction.Set(verticesx2[i], verticesy2[i], (i < 4) ? -5 : 5);
    Distref = direction.Mag();
    direction.Normalize();
    Dist = trap2.DistanceToOut(pzero, direction);
    // std::cout << "Dist=" << Dist << "  Distref=" << Distref << std::endl;
    VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, Distref));
  }

  // SafetyToIn(P)

  Dist = trap2.SafetyToIn(pbigx);
  VECGEOM_ASSERT(Dist <= 97.);
  Dist = trap2.SafetyToIn(pbigy);
  VECGEOM_ASSERT(Dist <= 97.);
  Dist = trap1.SafetyToIn(pbigz);
  VECGEOM_ASSERT(Dist <= 95.);
  Dist = trap1.SafetyToIn(ponzm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));
  Dist = trap2.SafetyToIn(pzero);
  VECGEOM_ASSERT(Dist < 0.);

  Vec_t testp;
  Precision testValue = 0.15;
  // X
  valid = trap1.Normal(ponxp, normal);
  testp = ponxp + testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist <= testValue);

  testp = ponxp - testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist < 0);

  valid = trap1.Normal(ponxm, normal);
  testp = ponxm + testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist <= testValue);

  testp = ponxm - testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist < 0);

  // Y
  valid = trap1.Normal(ponyp, normal);
  testp = ponyp + testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist <= testValue);

  testp = ponyp - testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist < 0);

  valid = trap1.Normal(ponym, normal);
  testp = ponym + testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist <= testValue);

  testp = ponym - testValue * normal;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist < 0);
  // Z
  valid = trap2.Normal(ponzp, normal);
  testp = ponzp + testValue * normal;
  Dist  = trap2.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponzp - testValue * normal;
  Dist  = trap2.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist < 0);

  valid = trap2.Normal(ponzm, normal);
  testp = ponzm + testValue * normal;
  Dist  = trap2.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponzm - testValue * normal;
  Dist  = trap2.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist < 0);

  // DistanceToIn(P,V)
  Vec_t dir;
  dir       = ponxm - pbigmx;
  testValue = dir.Mag();
  dir.Normalize();
  Dist = trap1.DistanceToIn(pbigmx, dir);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  dir       = ponxp - pbigx;
  testValue = dir.Mag();
  dir.Normalize();
  Dist = trap1.DistanceToIn(pbigx, dir);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  dir       = ponym - pbigmy;
  testValue = dir.Mag();
  dir.Normalize();
  Dist = trap1.DistanceToIn(pbigmy, dir);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  dir       = ponyp - pbigy;
  testValue = dir.Mag();
  dir.Normalize();
  Dist = trap1.DistanceToIn(pbigy, dir);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  // SamplePointOnSurface + DistanceToIn
  // Shoot from outside to points on surface
  for (int i = 0; i < 100; i++) {
    Vec_t psurf = trap2.GetUnplacedVolume()->SamplePointOnSurface();
    Vec_t start = 100. * psurf;
    dir         = psurf - start;
    testValue   = dir.Mag();
    dir.Normalize();
    Dist = trap2.DistanceToIn(start, dir);
    //      std::cout << "point: " << start << " dir: " << dir << " Dist=" << Dist << "  testValue=" << testValue <<
    //      "\n";
    VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
    Dist = trap2.DistanceToIn(psurf, -dir);
    VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  }


  // Check Extent and cached BBox

  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  trap2.Extent(minExtent, maxExtent);
  trap2.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-3, -3, -5)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(3, 3, 5)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));
  return true;
}

int main(int argc, char *argv[])
{
  TestGenTrap();
  std::cout << "VecGeom GenTrap passed\n";
  return 0;
}
