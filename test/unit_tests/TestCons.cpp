//
// File: TestCons.cpp
// Purpose: Unit tests for cones and cone sections

//  Ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "ApproxEqual.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/base/Global.h"
#include <cmath>
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/management/GeoManager.h"

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

#define DELTA 0.0001

using namespace vecgeom;
bool testingvecgeom = true;

// Returns false if actual is within wanted+/- DELTA
//         true if error
bool OutRange(Precision actual, Precision wanted)
{
  bool rng = false;
  if (actual < wanted - DELTA || actual > wanted + DELTA) {
    rng = true;
  }
  return rng;
}

bool OutRange(Vector3D<Precision> actual, Vector3D<Precision> wanted)
{
  bool rng = false;
  if (OutRange(actual.x(), wanted.x()) || OutRange(actual.y(), wanted.y()) || OutRange(actual.z(), wanted.z())) {
    rng = true;
  }
  return rng;
}

template <class Cone_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestCons()
{
  Vec_t pzero(0, 0, 0);

  Vec_t pplx(120, 0, 0), pply(0, 120, 0), pplz(0, 0, 120);

  Vec_t pmix(-120, 0, 0), pmiy(0, -120, 0), pmiz(0, 0, -120);

  Vec_t ponmiz(0, 75, -50), ponplz(0, 75, 50);

  Vec_t ponr1(std::sqrt(50 * 50 / 2.0), std::sqrt(50 * 50 / 2.0), 0);

  Vec_t ponr2(std::sqrt(100 * 100 / 2.0), std::sqrt(100 * 100 / 2.0), 0);

  Vec_t ponphi1(60 * std::cos(VECGEOM_NAMESPACE::kPi / 6), -60 * std::sin(VECGEOM_NAMESPACE::kPi / 6), 0);

  Vec_t ponphi2(60 * std::cos(VECGEOM_NAMESPACE::kPi / 6), 60 * std::sin(VECGEOM_NAMESPACE::kPi / 6), 0);

  Vec_t ponr2b(150, 0, 0);

  Vec_t pnearplz(45, 45, 45), pnearmiz(45, 45, -45);
  Vec_t pydx(60, 150, 0), pbigx(500, 0, 0);

  Vec_t proot1(0, 125, -1000), proot2(0, 75, -1000);

  Vec_t pparr1(0, 25, -150); // Test case parallel to both rs of c8
  Vec_t pparr2(0, 75, -50), pparr3(0, 125, 50);
  Vec_t vparr(0, 1. / std::sqrt(5.), 2. / std::sqrt(5.));

  Vec_t vnphi1(-std::sin(VECGEOM_NAMESPACE::kPi / 6), -std::cos(VECGEOM_NAMESPACE::kPi / 6), 0),
      vnphi2(-std::sin(VECGEOM_NAMESPACE::kPi / 6), std::cos(VECGEOM_NAMESPACE::kPi / 6), 0);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1), vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1),
      vxy(1. / std::sqrt(2.), 1. / std::sqrt(2.), 0), vxmy(1. / std::sqrt(2.), -1. / std::sqrt(2.), 0),
      vmxmy(-1. / std::sqrt(2.), -1. / std::sqrt(2.), 0), vmxy(-1. / std::sqrt(2.), 1. / std::sqrt(2.), 0),
      vx2mz(1.0 / std::sqrt(5.0), 0, -2.0 / std::sqrt(5.0)), vxmz(1. / std::sqrt(2.), 0, -1. / std::sqrt(2.));

  Cone_t c1("c1 Hollow Full Tube", 50, 100, 50, 100, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      cn1("cn1", 45., 50., 45., 50., 50, 0.5 * VECGEOM_NAMESPACE::kPi, 0.5 * VECGEOM_NAMESPACE::kPi),
      cn2("cn2", 45., 50., 45., 50., 50, 0.5 * VECGEOM_NAMESPACE::kPi, 1.5 * VECGEOM_NAMESPACE::kPi),
      c2("c2 Hollow Full Cone", 50, 100, 50, 200, 50, -1, 2 * VECGEOM_NAMESPACE::kPi),
      c3("c3 Hollow Cut Tube", 50, 100, 50, 100, 50, -VECGEOM_NAMESPACE::kPi / 6, VECGEOM_NAMESPACE::kPi / 3),
      c4("c4 Hollow Cut Cone", 50, 100, 50, 200, 50, -VECGEOM_NAMESPACE::kPi / 6, VECGEOM_NAMESPACE::kPi / 3),
      c5("c5 Hollow Cut Cone", 25, 50, 75, 150, 50, 0, 1.5 * VECGEOM_NAMESPACE::kPi),
      c6("c6 Solid Full Tube", 0, 150, 0, 150, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      c7("c7 Thin Tube", 95, 100, 95, 100, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      c8a("c8a Solid Full Cone2", 0, 100, 0, 150, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      c8b("c8b Hollow Full Cone2", 50, 100, 100, 150, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      c8c("c8c Hollow Full Cone2inv", 100, 150, 50, 100, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      c9("c9 Exotic Cone", 50, 60,
         0, // 1.0e-7,   500*kRadTolerance,
         10, 50, 0, 2 * VECGEOM_NAMESPACE::kPi),
      cms("cms cone", 0.0, 70.0, 0.0, 157.8, 2949.0, 0.0, 6.283185307179586);

  Cone_t cms2("RearAirCone", 401.0, 1450.0, 1020.0, 1450.0, 175.0, 0.0, 6.283185307179586);
  Cone_t ctest10("ctest10 cone", 20., 60., 80., 140., 100., 10 * VECGEOM_NAMESPACE::kPi / 180.,
                 300 * VECGEOM_NAMESPACE::kPi / 180.);

  Vec_t pct10(60, 0, 0);
  Vec_t pct10mx(-50, 0, 0);
  Vec_t pct10phi1(60 * std::cos(10. * VECGEOM_NAMESPACE::kPi / 180.), 60 * std::sin(10 * VECGEOM_NAMESPACE::kPi / 180.),
                  0);
  Vec_t pct10phi2(60 * std::cos(50. * VECGEOM_NAMESPACE::kPi / 180.),
                  -60 * std::sin(50 * VECGEOM_NAMESPACE::kPi / 180.), 0);

  Vec_t pct10e1(-691 - 500, 174, 404);

  Vec_t pct10e2(400 - 500, 20.9, 5.89);

  Vec_t pct10e3(456 - 500, 13, -14.7);

  Vec_t pct10e4(537 - 500, 1.67, -44.1);
  // point P is outside
  Vec_t pct10e5(537, 1.67, -44.1);

  Vec_t pct10e6(1e+03, -63.5, -213);
  Precision tolerance = vecgeom::kTolerance * 1000;

  Precision a1, a2, a3, am;

  a1 = pct10e2.x() - pct10e1.x();
  a2 = pct10e2.y() - pct10e1.y();
  a3 = pct10e2.z() - pct10e1.z();
  am = std::sqrt(a1 * a1 + a2 * a2 + a3 * a3);
  Vec_t d1(a1 / am, a2 / am, a3 / am);
  // std::cout<<d1.x()<<"\t"<<d1.y()<<"\t"<<d1.z()<<std::endl;

  a1 = pct10e3.x() - pct10e2.x();
  a2 = pct10e3.y() - pct10e2.y();
  a3 = pct10e3.z() - pct10e2.z();
  am = std::sqrt(a1 * a1 + a2 * a2 + a3 * a3);
  Vec_t d2(a1 / am, a2 / am, a3 / am);
  // std::cout<<d2.x()<<"\t"<<d2.y()<<"\t"<<d2.z()<<std::endl;

  a1 = pct10e4.x() - pct10e3.x();
  a2 = pct10e4.y() - pct10e3.y();
  a3 = pct10e4.z() - pct10e3.z();
  am = std::sqrt(a1 * a1 + a2 * a2 + a3 * a3);
  Vec_t d3(a1 / am, a2 / am, a3 / am);
  // std::cout<<d3.x()<<"\t"<<d3.y()<<"\t"<<d3.z()<<std::endl;

  // 19.01.04 modified test10 info:

  Vec_t pt10s1(6.454731216775542, -90.42080754048007, 100.);

  Vec_t pt10s2(22.65282328600368, -69.34877585931267, 76.51600623610082);

  Vec_t pt10s3(51.28206938732319, -32.10510677306267, 35.00932544708616);

  Vec_t vt10d(0.4567090876640433, 0.5941309830320264, -0.6621368319663807);

  Vec_t norm;

  // Check name
  // VECGEOM_ASSERT(c1.GetName()=="c1 Hollow Full Tube");

  // Check Cubic volume
  Precision vol, volCheck;
  vol      = c1.Capacity();
  volCheck = 2 * VECGEOM_NAMESPACE::kPi * 50 * (100 * 100 - 50 * 50);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  vol      = c6.Capacity();
  volCheck = 2 * VECGEOM_NAMESPACE::kPi * 50 * (150 * 150);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check Surface area
  vol      = c1.SurfaceArea();
  volCheck = 2 * VECGEOM_NAMESPACE::kPi * (50 * 2 * 50 + 100 * 2 * 50 + 100 * 100 - 50 * 50);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check Inside
  vecgeom::EnumInside in;
  std::cout.precision(16);
  // std::cout << "Testing Cone_t::Inside...\n";

  in = ctest10.Inside(pct10e1);
  // std::cout << "ctest10.Inside(pct10e1) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10e2);
  // std::cout << "ctest10.Inside(pct10e2) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kInside);

  in = ctest10.Inside(pct10e3);
  // std::cout << "ctest10.Inside(pct10e3) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kInside);

  in = ctest10.Inside(pct10e4);
  // std::cout << "ctest10.Inside(pct10e4) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10e5);
  // std::cout << "ctest10.Inside(pct10e5) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10e6);
  // std::cout << "ctest10.Inside(pct10e6) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10mx);
  // std::cout << "ctest10.Inside(pct10mx) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kSurface);

  in = ctest10.Inside(pt10s1);
  // std::cout << "ctest10.Inside(pt10s1) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kSurface);

  in = ctest10.Inside(pt10s2);
  // std::cout << "ctest10.Inside(pt10s2) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kSurface);

  in = ctest10.Inside(pt10s3);
  // std::cout << "ctest10.Inside(pt10s3) = " <<in<< std::endl;
  VECGEOM_ASSERT(in == vecgeom::EInside::kOutside);

  vecgeom::Inside_t aux;
  if ((aux = c1.Inside(pzero)) != vecgeom::EInside::kOutside)
    std::cout << "c1.Inside() mismatch: Line " << __LINE__ << ", p=" << pzero << ", enumInside=" << c1.Inside(pzero)
              << "\n";
  if ((aux = c6.Inside(pzero)) != vecgeom::EInside::kInside)
    std::cout << "c6.Inside() mismatch: Line " << __LINE__ << ", p=" << pzero << ", enumInside=" << aux << "\n";
  if ((aux = c1.Inside(pplx)) != vecgeom::EInside::kOutside)
    std::cout << "c1.Inside() mismatch: Line " << __LINE__ << ", p=" << pplx << ", enumInside=" << aux << "\n";
  if ((aux = c2.Inside(pplx)) != vecgeom::EInside::kInside)
    std::cout << "c2.Inside() mismatch: Line " << __LINE__ << ", p=" << pplx << ", enumInside=" << aux << "\n";
  if ((aux = c3.Inside(pplx)) != vecgeom::EInside::kOutside)
    std::cout << "c3.Inside() mismatch: Line " << __LINE__ << ", p=" << pplx << ", enumInside=" << aux << "\n";
  if ((aux = c4.Inside(pplx)) != vecgeom::EInside::kInside)
    std::cout << "c4.Inside() mismatch: Line " << __LINE__ << ", p=" << pplx << ", enumInside=" << aux << "\n";
  if ((aux = c1.Inside(ponmiz)) != vecgeom::EInside::kSurface)
    std::cout << "c1.Inside() mismatch: Line " << __LINE__ << ", p=" << ponmiz << ", enumInside=" << aux << "\n";
  if ((aux = c1.Inside(ponplz)) != vecgeom::EInside::kSurface)
    std::cout << "c1.Inside() mismatch: Line " << __LINE__ << ", p=" << ponplz << ", enumInside=" << aux << "\n";
  if ((aux = c1.Inside(ponr1)) != vecgeom::EInside::kSurface)
    std::cout << "c1.Inside() mismatch: Line " << __LINE__ << ", p=" << ponr1 << ", enumInside=" << aux << "\n";
  if ((aux = c1.Inside(ponr2)) != vecgeom::EInside::kSurface)
    std::cout << "c1.Inside() mismatch: Line " << __LINE__ << ", p=" << ponr2 << ", enumInside=" << aux << "\n";
  if ((aux = c3.Inside(ponphi1)) != vecgeom::EInside::kSurface)
    std::cout << "c3.Inside() mismatch: Line " << __LINE__ << ", p=" << ponphi1 << ", enumInside=" << aux << "\n";
  if ((aux = c3.Inside(ponphi2)) != vecgeom::EInside::kSurface)
    std::cout << "c3.Inside() mismatch: Line " << __LINE__ << ", p=" << ponphi2 << ", enumInside=" << aux << "\n";

  if ((aux = c5.Inside(Vec_t(70, 1, 0))) != vecgeom::EInside::kInside)
    std::cout << "c5.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(10, 1, 0) << ", enumInside=" << aux
              << "\n";
  if ((aux = c5.Inside(Vec_t(50, -50, 0))) != vecgeom::EInside::kOutside)
    std::cout << "c5.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(50, -50, 0) << ", enumInside=" << aux
              << "\n";
  if ((aux = c5.Inside(Vec_t(70, 0, 0))) != vecgeom::EInside::kSurface)
    std::cout << "c5.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(70, 0, 0) << ", enumInside=" << aux
              << "\n";
  // on tolerant r, inside z, within phi
  if ((aux = c5.Inside(Vec_t(100, 0, 0))) != vecgeom::EInside::kSurface)
    std::cout << "c5.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(100, 0, 0) << ", enumInside=" << aux
              << "\n";
  if ((aux = c3.Inside(Vec_t(100, 0, 0))) != vecgeom::EInside::kSurface)
    std::cout << "c3.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(100, 0, 0) << ", enumInside=" << aux
              << "\n";
  // on tolerant r, tolerant z, within phi
  if ((aux = c5.Inside(Vec_t(100, 0, 50))) != vecgeom::EInside::kSurface)
    std::cout << "c5.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(100, 0, 50) << ", enumInside=" << aux
              << "\n";
  if ((aux = c3.Inside(Vec_t(100, 0, 50))) != vecgeom::EInside::kSurface)
    std::cout << "c3.Inside() mismatch: Line " << __LINE__ << ", p=" << Vec_t(100, 0, 50) << ", enumInside=" << aux
              << "\n";

  Precision p2 = 1. / std::sqrt(2.), p3 = 1. / std::sqrt(3.);
  bool valid;

  valid = cn1.Normal(Vec_t(0., 50., 0.), norm);
  if (OutRange(norm, Vec_t(p2, p2, 0)))
    std::cout << "cn1.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0, 50, 0) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(p2, p2, 0.)) && valid);
  valid = cn1.Normal(Vec_t(0., 45., 0.), norm);
  if (OutRange(norm, Vec_t(p2, -p2, 0)))
    std::cout << "cn1.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0, 45, 0) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(p2, -p2, 0.)));
  valid = cn1.Normal(Vec_t(0., 45., 50.), norm);
  if (OutRange(norm, Vec_t(p3, -p3, p3)))
    std::cout << "cn1.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0, 45, 50) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(p3, -p3, p3)));
  valid = cn1.Normal(Vec_t(0., 45., -50.), norm);
  if (OutRange(norm, Vec_t(p3, -p3, -p3)))
    std::cout << "cn1.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0, 45, -50) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(p3, -p3, -p3)));
  valid = cn1.Normal(Vec_t(-50., 0., -50.), norm);
  if (OutRange(norm, Vec_t(-p3, -p3, -p3)))
    std::cout << "cn1.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(-50, 0, -50) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(-p3, -p3, -p3)));
  valid = cn1.Normal(Vec_t(-50., 0., 0.), norm);
  if (OutRange(norm, Vec_t(-p2, -p2, 0)))
    std::cout << "cn1.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(-50, 0, 0) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(-p2, -p2, 0.)));
  valid = cn2.Normal(Vec_t(50., 0., 0.), norm);
  if (OutRange(norm, Vec_t(p2, p2, 0)))
    std::cout << "cn2.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(50, 0, 0) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(p2, p2, 0.)));
  valid = c6.Normal(Vec_t(0., 0., 50.), norm);
  if (OutRange(norm, Vec_t(0, 0, 1)))
    std::cout << "c6.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0, 0, 50) << ", valid=" << valid
              << ", normal=" << norm << "\n";
  VECGEOM_ASSERT(ApproxEqual(norm, Vec_t(0., 0., 1.)));

  valid = c1.Normal(ponplz, norm);
  if (OutRange(norm, Vec_t(0, 0, 1)))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << ponplz << ", normal=" << norm << "\n";
  valid = c1.Normal(ponmiz, norm);
  if (OutRange(norm, Vec_t(0, 0, -1)))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << ponmiz << ", normal=" << norm << "\n";
  valid = c1.Normal(ponr1, norm);
  // if (OutRange(norm,Vec_t(-1.0/std::sqrt(2.0),-1.0/std::sqrt(2.0),0)))
  if (OutRange(norm, Vec_t(-p2, -p2, 0)))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << ponr1 << ", normal=" << norm << "\n";
  valid = c1.Normal(ponr2, norm);
  if (OutRange(norm, Vec_t(p2, p2, 0)))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << ponr2 << ", normal=" << norm << "\n";
  valid = c3.Normal(ponphi1, norm);
  if (OutRange(norm, vnphi1))
    std::cout << "c3.Normal() mismatch: Line " << __LINE__ << ", p=" << ponphi1 << ", normal=" << norm << "\n";
  valid = c3.Normal(ponphi2, norm);
  if (OutRange(norm, vnphi2))
    std::cout << "c3.Normal() mismatch: Line " << __LINE__ << ", p=" << ponphi2 << ", normal=" << norm << "\n";
  valid = c4.Normal(ponr2b, norm);
  if (OutRange(norm, vxmz))
    std::cout << "c4.Normal() mismatch: Line " << __LINE__ << ", p=" << ponr2b << ", normal=" << norm << "\n";

  valid = c5.Normal(Vec_t(0.5, 0, -50), norm);
  if (OutRange(norm, Vec_t(0., 0., 0.)))
    std::cout << "c5.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0.5, 0, -50) << ", normal=" << norm
              << ", valid=" << valid << "\n";

  valid = c5.Normal(Vec_t(500, 0, -50), norm);
  if (OutRange(norm, Vec_t(0., 0., 0.)))
    std::cout << "c5.Normal() mismatch: Line " << __LINE__ << ", p=" << Vec_t(500, 0, -50) << ", normal=" << norm
              << ", valid=" << valid << "\n";

  // check that Normal() returns (0,0,0) for points away from the surface

  if ((valid = c1.Normal(pplz, norm)) || OutRange(norm, pzero))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << pplz << ", normal=" << norm
              << ", valid=" << valid << "\n";
  if ((valid = c1.Normal(pmiz, norm)) || OutRange(norm, pzero))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << pmiz << ", normal=" << norm
              << ", valid=" << valid << "\n";
  if ((valid = c1.Normal(pbigx, norm)) || OutRange(norm, pzero))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << pbigx << ", normal=" << norm
              << ", valid=" << valid << "\n";
  if ((valid = c1.Normal(pzero, norm)) || OutRange(norm, pzero))
    std::cout << "c1.Normal() mismatch: Line " << __LINE__ << ", p=" << pzero << ", normal=" << norm
              << ", valid=" << valid << "\n";

  // safety

  Precision dist;
  dist = c4.SafetyToOut(ponphi1);
  if (OutRange(dist, 0))
    std::cout << "c4.S2O() mismatch: Line " << __LINE__ << ", p=" << ponphi1 << ", dist=" << dist << "\n";

  dist = c1.SafetyToOut(ponphi1);
  if (OutRange(dist, 10))
    std::cout << "c1.S2O() mismatch: Line " << __LINE__ << ", p=" << ponphi1 << ", dist=" << dist << "\n";

  dist = c1.SafetyToOut(pnearplz);
  if (OutRange(dist, 5))
    std::cout << "c1.S2O() mismatch: Line " << __LINE__ << ", p=" << pnearplz << ", dist=" << dist << "\n";

  dist = c1.SafetyToOut(pnearmiz);
  if (OutRange(dist, 5))
    std::cout << "c1.S2O() mismatch: Line " << __LINE__ << ", p=" << pnearmiz << ", dist=" << dist << "\n";

  dist = c1.SafetyToOut(ponr1);
  if (OutRange(dist, 0))
    std::cout << "c1.S2O() mismatch: Line " << __LINE__ << ", p=" << ponr1 << ", dist=" << dist << "\n";
  dist = c1.SafetyToOut(ponr2);
  if (OutRange(dist, 0))
    std::cout << "c1.S2O() mismatch: Line " << __LINE__ << ", p=" << ponr2 << ", dist=" << dist << "\n";

  dist = c6.SafetyToOut(pzero);
  if (OutRange(dist, 50))
    std::cout << "c6.S2O() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=" << dist << "\n";

  dist = c5.SafetyToOut(Vec_t(0, -70, 0));
  if (OutRange(dist, 0))
    std::cout << "c5.S2O() mismatch: Line " << __LINE__ << ", p=" << Vec_t(0, -70, 0) << ", dist=" << dist << "\n";

  // distToOut

  dist = c4.DistanceToOut(pplx, vx);
  if (OutRange(dist, 30) || OutRange(norm, vxmz))
    std::cout << "c4.D2O() mismatch: Line " << __LINE__ << ", p=" << pplx << ", dir=" << vx << ", dist=" << dist << " "
              << norm << "\n";

  dist = c2.DistanceToOut(pplx, vx);
  if (OutRange(dist, 30) || OutRange(norm, vxmz))
    std::cout << "c2.D2O() mismatch: Line " << __LINE__ << ", p=" << pplx << ", dir=" << vx << ", dist=" << dist << " "
              << norm << "\n";

  dist = c4.DistanceToOut(pplx, vmx);
  if (OutRange(dist, 70))
    std::cout << "c4.D2O() mismatch: Line " << __LINE__ << ", p=" << pplx << ", dir=" << vmx << ", dist=" << dist << " "
              << norm << "\n";

  dist = c2.DistanceToOut(pplx, vmx);
  if (OutRange(dist, 70))
    std::cout << "c2.D2O() mismatch: Line " << __LINE__ << ", p=" << pplx << ", dir=" << vmx << ", dist=" << dist << " "
              << norm << "\n";

  dist = c3.DistanceToOut(ponphi1, vmy);
  if (OutRange(dist, 0) || OutRange(norm, vnphi1))
    std::cout << "c3.D2O() mismatch: Line " << __LINE__ << ", PhiS1 p=" << ponphi1 << ", dir=" << vmy
              << ", dist=" << dist << " normal=" << norm << "\n";
  dist = c3.DistanceToOut(ponphi1, vy);
  // norm=pNorm->unit();
  if (OutRange(dist, 2 * 60 * std::sin(VECGEOM_NAMESPACE::kPi / 6)) || OutRange(norm, vnphi2))
    std::cout << "c3.D2O() mismatch: Line " << __LINE__ << ", PhiS2 p=" << ponphi1 << ", dir=" << vy
              << ", dist=" << dist << " " << norm << "\n";

  dist = c3.DistanceToOut(ponphi2, vy);
  if (OutRange(dist, 0) || OutRange(norm, vnphi2))
    std::cout << "c3.D2O() mismatch: Line " << __LINE__ << ", PhiE1 p=" << ponphi2 << ", dir=" << vy
              << ", dist=" << dist << " " << norm << "\n";

  dist = c3.DistanceToOut(ponphi2, vmy);
  if (OutRange(dist, 2 * 60 * std::sin(VECGEOM_NAMESPACE::kPi / 6)) || OutRange(norm, vnphi1))
    std::cout << "c3.D2O() mismatch: Line " << __LINE__ << ", PhiE3 p=" << ponphi2 << ", dir=" << vmy
              << ", dist=" << dist << " " << norm << "\n";

  dist = c6.DistanceToOut(ponplz, vmz);
  if (OutRange(dist, 100) || OutRange(norm, vmz))
    std::cout << "c6.D2O() mismatch: Line " << __LINE__ << ", Top Z1 p=" << ponplz << ", dir=" << vmz
              << ", dist=" << dist << " " << norm << "\n";
  dist = c6.DistanceToOut(ponplz, vz);
  if (OutRange(dist, 0) || OutRange(norm, vz))
    std::cout << "c6.D2O() mismatch: Line " << __LINE__ << ", Top Z2 p=" << ponplz << ", dir=" << vz
              << ", dist=" << dist << " " << norm << "\n";

  dist = c6.DistanceToOut(ponmiz, vz);
  if (OutRange(dist, 100) || OutRange(norm, vz))
    std::cout << "c6.D2O() mismatch: Line " << __LINE__ << ", Lower Z1 p=" << ponmiz << ", dir=" << vz
              << ", dist=" << dist << " " << norm << "\n";

  dist = c6.DistanceToOut(ponmiz, vmz);
  if (OutRange(dist, 0) || OutRange(norm, vmz))
    std::cout << "c6.D2O() mismatch: Line " << __LINE__ << ", Lower z2 p=" << ponmiz << ", dir=" << vmz
              << ", dist=" << dist << " " << norm << "\n";

  // Test case for rmax root bug
  dist           = c7.DistanceToOut(ponr2, vmx);
  Precision aux1 = 100. / std::sqrt(2.) - std::sqrt(95. * 95. - 100. * 100. / 2.);
  if (OutRange(dist, aux1))
    std::cout << "c7.D2O() mismatch: Line " << __LINE__ << ", rmax root bug p=" << ponr2 << ", dir=" << vmx
              << ", dist=" << dist << " " << norm << "\n";

  // Parallel radii test cases
  dist = c8a.DistanceToOut(pparr2, vparr);
  if (OutRange(dist, 100. * std::sqrt(5.) / 2.) || OutRange(norm, vz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr2a p=" << pparr2 << ", dir=" << vparr
              << ", dist=" << dist << " " << norm << "\n";

  dist = c8a.DistanceToOut(pparr2, -vparr);
  if (OutRange(dist, 0) || OutRange(norm, vmz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr2b p=" << pparr2 << ", dir=" << -vparr
              << ", dist=" << dist << " " << norm << "\n";

  dist = c8a.DistanceToOut(pparr2, vz);
  if (OutRange(dist, 100) || OutRange(norm, vz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr2c p=" << pparr2 << ", dir=" << vz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8a.DistanceToOut(pparr2, vmz);
  if (OutRange(dist, 0) || OutRange(norm, vmz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr2d p=" << pparr2 << ", dir=" << vmz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8a.DistanceToOut(pparr3, vparr);
  if (OutRange(dist, 0) || OutRange(norm, vz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr3a p=" << pparr3 << ", dir=" << vparr
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8a.DistanceToOut(pparr3, -vparr);
  if (OutRange(dist, 100 * std::sqrt(5.) / 2.) || OutRange(norm, vmz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr3b p=" << pparr3 << ", dir=" << -vparr
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8a.DistanceToOut(pparr3, vz);
  if (OutRange(dist, 0) || OutRange(norm, vz))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", solid parr3c p=" << pparr3 << ", dir=" << vz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8a.DistanceToOut(pparr3, vmz);
  if (OutRange(dist, 50) || OutRange(norm, Vec_t(0, 2. / std::sqrt(5.0), -1. / std::sqrt(5.0))))
    std::cout << "c8a.D2O() mismatch: Line " << __LINE__ << ", p=" << pparr3 << ", dir=" << vmz << ", dist=" << dist
              << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr2, vparr);
  if (OutRange(dist, 100 * std::sqrt(5.) / 2.) || OutRange(norm, vz))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr2a p=" << pparr2 << ", dir=" << vparr
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr2, -vparr);
  if (OutRange(dist, 0) || OutRange(norm, vmz))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr2b p=" << pparr2 << ", dir=" << -vparr
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr2, vz);
  if (OutRange(dist, 50))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr2c p=" << pparr2 << ", dir=" << vz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr2, vmz);
  if (OutRange(dist, 0) || OutRange(norm, vmz))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr2d p=" << pparr2 << ", dir=" << vmz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr3, vparr);
  if (OutRange(dist, 0) || OutRange(norm, vz))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr3 p=" << pparr3 << ", dir=" << vparr
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr3, -vparr);
  if (OutRange(dist, 100. * std::sqrt(5.) / 2.) || OutRange(norm, vmz))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr3b p=" << pparr3 << ", dir=" << -vparr
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr3, vz);
  if (OutRange(dist, 0) || OutRange(norm, vz))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr3c p=" << pparr3 << ", dir=" << vz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c8b.DistanceToOut(pparr3, vmz);
  if (OutRange(dist, 50) || OutRange(norm, Vec_t(0, 2. / std::sqrt(5.), -1.0 / std::sqrt(5.))))
    std::cout << "c8b.D2O() mismatch: Line " << __LINE__ << ", hollow parr3d p=" << pparr3 << ", dir=" << vmz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c9.DistanceToOut(Vec_t(1e3 * tolerance, 0, 50), vx2mz);
  if (testingvecgeom) {
    if (OutRange(dist, 111.8033988))
      std::cout << "c9.D2Out() mismatch: Line " << __LINE__ << ", p=(1e3*tolerance,0,50), dir=" << vx2mz
                << ", dist=" << dist << " normal=" << norm << "\n";
  } else {
    if (OutRange(dist, 111.8033988) || OutRange(norm, Vec_t(0, 0, -1.0)))
      std::cout << "c9.D2Out() mismatch: Line " << __LINE__ << ", p=(1e3*tolerance,0,50), dir=" << vx2mz
                << ", dist=" << dist << " normal=" << norm << "\n";
  }
  dist = c9.DistanceToOut(Vec_t(5, 0, 50), vx2mz);
  if (OutRange(dist, 111.8033988) || OutRange(norm, Vec_t(0, 0, -1.0)))
    std::cout << "c9.D2O() mismatch: Line " << __LINE__ << ", p=" << Vec_t(5, 0, 50) << ", dir=" << vx2mz
              << ", dist=" << dist << " normal=" << norm << "\n";

  dist = c9.DistanceToOut(Vec_t(10, 0, 50), vx2mz);
  if (testingvecgeom) {
    if (OutRange(dist, 111.8033988))
      std::cout << "c9.D2O() mismatch: Line " << __LINE__ << ", p=(10,0,50), dir=" << vx2mz << ", dist=" << dist
                << " normal=" << norm << "\n";
  } else {
    if (OutRange(dist, 111.8033988) || OutRange(norm, Vec_t(0, 0, -1.0)))
      std::cout << "c9.D2O() mismatch: Line " << __LINE__ << ", p=(10,0,50), dir=" << vx2mz << ", dist=" << dist
                << " normal=" << norm << "\n";
  }

  Vec_t point = {0.28628920024909, -0.43438111004815, -2949.0};
  Vec_t dir   = {6.0886686196674e-05, -9.2382200635766e-05, 0.99999999387917};
  dist        = cms.DistanceToOut(point, dir);
  if (OutRange(dist, 5898.0))
    std::cout << "cms.D2O() mismatch: Line " << __LINE__ << ", p=" << point << ", dir=" << dir << ", dist=" << dist
              << " normal=" << norm << "\n";

  point = Vec_t(0.28628920024909, -0.43438111004815, -2949.0 + tolerance * 0.25);
  dir   = Vec_t(6.0886686196674e-05, -9.2382200635766e-05, 0.99999999387917);
  dist  = cms.DistanceToOut(point, dir);
  if (OutRange(dist, 5898.0))
    std::cout << "cms.D2O() mismatch: Line " << __LINE__ << ", p=" << point << ", dir=" << dir << ", dist=" << dist
              << " normal=" << norm << "\n";

  point = Vec_t(0.28628920024909, -0.43438111004815, -2949.0 - tolerance * 0.25);
  dir   = Vec_t(6.0886686196674e-05, -9.2382200635766e-05, 0.99999999387917);
  dist  = cms.DistanceToOut(point, dir);
  if (OutRange(dist, 5898.0))
    std::cout << "cms.D2O() mismatch: Line " << __LINE__ << ", p=" << point << ", dir=" << dir << ", dist=" << dist
              << " " << norm << "\n";

  point = Vec_t(-344.13684353113, 258.98049377272, -158.20772167926);
  dir   = Vec_t(-0.30372024336672, -0.5581146924652, 0.77218003329776);
  dist  = cms2.DistanceToOut(point, dir);
  if (OutRange(dist, 0.))
    std::cout << "cms.D2O() mismatch: Line " << __LINE__ << ", p=" << point << ", dir=" << dir << ", dist=" << dist
              << " " << norm << "\n";

  dist = ctest10.DistanceToOut(pct10e2, d1);
  // if (OutRange(dist,111.8033988)||
  //     convex) //||
  //        OutRange(norm,Vec_t(0,0,-1.0))) this is false!
  // instead of checking an arbitrary value, check that point+step*direction is on the surface
  if (ctest10.Inside(pct10e2 + dist * d1) != vecgeom::EInside::kSurface)
    std::cout << "ctest10.Inside() mismatch: Line " << __LINE__ << ", p=" << pct10e2 + dist * d1 << ", p is on "
              << ctest10.Inside(pct10e2) << " pct10e2=" << pct10e2 << " norm=" << norm << "\n";

  dist = ctest10.DistanceToOut(pct10e3, d1);
  // norm=pNorm->unit();
  // if (OutRange(dist,111.8033988)||
  //     convex||
  //     OutRange(norm,Vec_t(0,0,-1.0)))
  if (ctest10.Inside(pct10e3 + dist * d1) != vecgeom::EInside::kSurface)
    std::cout << "ctest10.Inside() mismatch: Line " << __LINE__ << ", ctest10.DistanceToOut(pct10e3,d1,...) = " << dist
              << "\n";

  /////////////////////////////////////////////
  //

  // std::cout <<"Mismatch: Line "<< __LINE__ <<", Testing Cone_t::DistanceToIn(p) ...\n";

  dist = c1.SafetyToIn(pzero);
  if (OutRange(dist, 50))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=A " << dist << "\n";

  dist = c1.SafetyToIn(pplx);
  if (OutRange(dist, 20))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=B " << dist << "\n";

  dist = c1.SafetyToIn(pply);
  if (OutRange(dist, 20))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=C " << dist << "\n";

  dist = c4.SafetyToIn(pply);
  if (OutRange(dist, 120 * std::sin(VECGEOM_NAMESPACE::kPi / 3)))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=D " << dist << "\n";

  dist = c4.SafetyToIn(pmiy);
  if (OutRange(dist, 120 * std::sin(VECGEOM_NAMESPACE::kPi / 3)))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=D " << dist << "\n";

  dist = c1.SafetyToIn(pplz);
  if (OutRange(dist, 70))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=E " << dist << "\n";
  // Check with both rmins=0
  dist = c5.SafetyToIn(pplx);
  if (OutRange(dist, 20. / std::sqrt(2.)))
    std::cout << "S2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=F " << dist << "\n";

  /////////////////////////////////////////////////////
  //

  // std::cout <<"Mismatch: Line "<< __LINE__ <<", Testing Cone_t::DistanceToIn(p,v,...) ...\n";

  dist = c1.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=A " << dist << "\n";

  dist = c2.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c2.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c3.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c3.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c4.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c4.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c5.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c5.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c6.DistanceToIn(pplz, vmz);
  if (OutRange(dist, 70.0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c6.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c7.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c7.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c8a.DistanceToIn(pplz, vmz);
  if (OutRange(dist, 70.0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c8a.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c8b.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c8b.DistanceToIn(pplz,vmz) = " << dist << "\n";

  dist = c8c.DistanceToIn(pplz, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c8c.DistanceToIn(pplz,vmz) = " << dist << "\n";

  // Cone is with Rmin=Rmax=0 at +dz
  dist = c9.DistanceToIn(pplz, vmz);
  if (OutRange(dist, 70.0)) {
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c9.DistanceToIn(pplz,vmz) = " << dist << "\n";
  }
  dist = c9.DistanceToIn(Vec_t(0, 0, 50), vmz);
  if (OutRange(dist, 0.0)) {
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c9.DistanceToIn((0,0,50),vmz) = " << dist << "\n";
  }
  ///////////////

  dist = c1.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=A " << dist << "\n";

  dist = c2.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c2.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c3.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c3.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c4.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c4.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c5.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c5.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c6.DistanceToIn(pmiz, vz);
  if (OutRange(dist, 70.0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c6.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c7.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c7.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c8a.DistanceToIn(pmiz, vz);
  if (OutRange(dist, 70.0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c8a.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c8b.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c8b.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c8c.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c8c.DistanceToIn(pmiz,vz) = " << dist << "\n";

  dist = c9.DistanceToIn(pmiz, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", Error:c9.DistanceToIn(pmiz,vz) = " << dist << "\n";

  //////////////

  dist = c1.DistanceToIn(pplx, vmx);
  if (OutRange(dist, 20))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=B " << dist << "\n";
  dist = c1.DistanceToIn(pplz, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=C " << dist << "\n";
  dist = c4.DistanceToIn(pply, vmy);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=D " << dist << "\n";

  dist = c1.DistanceToIn(pydx, vmy);
  if (OutRange(dist, 70))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=E " << dist << "\n";
  dist = c3.DistanceToIn(pydx, vmy);
  if (OutRange(dist, 150 - 60 * std::tan(VECGEOM_NAMESPACE::kPi / 6)))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=F " << dist << "\n";

  dist = c1.DistanceToIn(pplx, vmx);
  if (OutRange(dist, 20))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=G " << dist << "\n";
  dist = c1.DistanceToIn(pplx, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=G2 " << dist << "\n";

  dist = c4.DistanceToIn(pbigx, vmx);
  if (OutRange(dist, 350))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=G3 " << dist << "\n";

  dist = c4.DistanceToIn(pzero, vx);
  if (OutRange(dist, 50))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=H " << dist << "\n";

  dist = c1.DistanceToIn(ponr2, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=I" << dist << "\n";
  dist = c1.DistanceToIn(ponr2, vmx);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=I2" << dist << "\n";

  dist = c1.DistanceToIn(ponr1, vx);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=J" << dist << "\n";
  dist = c1.DistanceToIn(ponr1, vmx);
  if (OutRange(dist, 2.0 * std::sqrt(50 * 50 / 2.)))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=J2" << dist << "\n";

  dist = c1.DistanceToIn(ponr2, vmxmy);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=K" << dist << "\n";

  // Parallel test case -> parallel to both radii
  dist = c8b.DistanceToIn(pparr1, vparr);
  if (OutRange(dist, 100 * std::sqrt(5.) / 2.))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=parr1 " << dist << "\n";
  dist = c8b.DistanceToIn(pparr2, -vparr);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=parr2 " << dist << "\n";
  dist = c8b.DistanceToIn(pparr3, vparr);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=parr3a " << dist << "\n";
  dist = c8b.DistanceToIn(pparr3, -vparr);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=parr3b " << dist << "\n";

  // Check we don't Hit `shadow cone' at `-ve radius' on rmax or rmin
  dist = c8a.DistanceToIn(proot1, vz);
  if (OutRange(dist, 1000))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=shadow rmax root problem " << dist
              << "\n";

  dist = c8c.DistanceToIn(proot2, vz);
  if (OutRange(dist, 1000))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", p=" << pzero << ", dist=shadow rmin root problem " << dist
              << "\n";

  dist = cms2.DistanceToIn(Vec_t(-344.13684353113, 258.98049377272, -158.20772167926),
                           Vec_t(-0.30372022869765, -0.55811472925794, 0.77218001247454));
  if (OutRange(dist, kInfLength)) std::cout << "cms2.DistanceToIn(Vec_t(-344.1 ... = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10,vx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10, vmx);
  if (OutRange(dist, 110))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10,vmx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10, vy);
  if (OutRange(dist, 10.57961))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10,vy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10, vmy);
  if (OutRange(dist, 71.5052))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10,vmy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10,vz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi1, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1,vx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi1, vmx);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1,vmx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi1, vy);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1,vy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi1, vmy);
  if (OutRange(dist, 80.83778))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1,vmy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi1, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1,vz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi1, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1,vmz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2,vx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2, vmx);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2,vmx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2, vy);
  if (OutRange(dist, 77.78352))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2,vy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2, vmy);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2,vmy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2,vz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2, vmz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2,vmz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10mx, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10mx,vx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10mx, vmx);
  if (OutRange(dist, 0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10mx,vmx) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10mx, vy);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10mx,vy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10mx, vmy);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10mx,vmy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10mx, vz);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10mx,vz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10mx, vmz);
  if (OutRange(dist, 0.0))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10mx,vmz) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10e1, d1);
  if (OutRange(dist, 1171.1712))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10e1,d1) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10e4, d1);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10e4,d1) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pt10s2, vt10d);
  if (OutRange(dist, 142.8017))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pt10s2,vt10d) = " << dist << "\n";

  Precision arad = 90.;

  Vec_t pct10phi1r(arad * std::cos(10. * VECGEOM_NAMESPACE::kPi / 180.),
                   arad * std::sin(10 * VECGEOM_NAMESPACE::kPi / 180.), 0);
  Vec_t pct10phi2r(arad * std::cos(50. * VECGEOM_NAMESPACE::kPi / 180.),
                   -arad * std::sin(50 * VECGEOM_NAMESPACE::kPi / 180.), 0);

  dist = ctest10.DistanceToIn(pct10phi1r, vmy);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi1r,vmy) = " << dist << "\n";

  dist = ctest10.DistanceToIn(pct10phi2r, vx);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(pct10phi2r,vx) = " << dist << "\n";

  Vec_t alex1P(49.840299921054168, -59.39735648688918, -20.893051766050633);
  Vec_t alex1V(0.6068108874999103, 0.35615926907657169, 0.71058505603651234);

  in = ctest10.Inside(alex1P);
  // std::cout <<"Inside() mismatch: Line "<< __LINE__ <<", ctest10.Inside(alex1P) = " <<in<<"\n";
  // VECGEOM_ASSERT(in == vecgeom::EInside::kSurface);

  dist = ctest10.DistanceToIn(alex1P, alex1V);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(alex1P,alex1V) = " << dist << "\n";

  dist = ctest10.DistanceToOut(alex1P, alex1V);
  if (OutRange(dist, 0))
    std::cout << "ctest10.D2O() mismatch: Line " << __LINE__ << ", p=" << alex1P << ", dir=" << alex1V
              << ", dist=" << dist << " " << norm << "\n";

  Vec_t alex2P(127.0075852717127, -514.1050841937065, 69.47104834263656);
  Vec_t alex2V(0.1277616879490939, 0.4093610465777845, 0.9033828007202369);

  in = ctest10.Inside(alex2P);
  // std::cout <<"Inside() mismatch: Line "<< __LINE__ <<", ctest10.Inside(alex2P) = " <<in<<"\n";
  VECGEOM_ASSERT(in == vecgeom::EInside::kOutside);

  dist = ctest10.DistanceToIn(alex2P, alex2V);
  if (OutRange(dist, kInfLength))
    std::cout << "D2I() mismatch: Line " << __LINE__ << ", ctest10.DistanceToIn(alex2P,alex2V) = " << dist << "\n";

  // Add Error of CMS, point on the Inner Surface going // to imaginary cone
  Cone_t testc("cCone", 261.9, 270.4, 1066.5, 1068.7, 274.75, 0., 2 * VECGEOM_NAMESPACE::kPi);
  dir = Vec_t(0.653315775, 0.5050862758, 0.5639737158);
  Precision x, y, z;
  x          = -296.7662086;
  y          = -809.1328836;
  z          = 13210.2270 - (12800.5 + 274.75);
  point      = Vec_t(x, y, z);
  dist       = testc.DistanceToOut(point, dir);
  Vec_t newp = point + dist * dir;
  // std::cout<<"CMS problem: DistOut has to be small="<<testc.DistanceToOut(point,dir,norm,convex)<<"\n";
  // std::cout<<"CMS problem: DistInNew has to be kInfLength="<<testc.DistanceToIn(newp,dir)<<"\n";
  VECGEOM_ASSERT(dist < 0.05);
  dist = testc.DistanceToIn(newp, dir);
  //  VECGEOM_ASSERT(ApproxEqual<Precision>(dist,kInfLength));

  // Second test for Cons derived from testG4Cons1.cc
  pbigx = Vec_t(100, 0, 0);
  Vec_t pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t ponxside(50, 0, 0);

  Precision Dist;

  Cone_t t1("Solid TubeLike #1", 0, 50, 0, 50, 50, 0, 2. * VECGEOM_NAMESPACE::kPi);
  Cone_t test10("test10", 20.0, 80.0, 60.0, 140.0, 100.0, 0.17453292519943, 5.235987755983);

  Cone_t test10a("aCone", 20, 60, 80, 140, 100, 10. * VECGEOM_NAMESPACE::kPi / 180.,
                 300. * VECGEOM_NAMESPACE::kPi / 180.);

  // Check Inside
  VECGEOM_ASSERT(t1.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(t1.Inside(pbigx) == vecgeom::EInside::kOutside);

  // Check Surface Normal

  valid = t1.Normal(ponxside, norm);
  VECGEOM_ASSERT(ApproxEqual(norm, vx));

  // SafetyToOut(P)
  Dist = t1.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));

  // DistanceToOut(P,V)
  Dist  = t1.DistanceToOut(pzero, vx);
  valid = t1.Normal(pzero + Dist * vx, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vx));
  Dist  = t1.DistanceToOut(pzero, vmx);
  valid = t1.Normal(pzero + Dist * vmx, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vmx));
  Dist  = t1.DistanceToOut(pzero, vy);
  valid = t1.Normal(pzero + Dist * vy, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vy));
  Dist  = t1.DistanceToOut(pzero, vmy);
  valid = t1.Normal(pzero + Dist * vmy, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vmy));
  Dist  = t1.DistanceToOut(pzero, vz);
  valid = t1.Normal(pzero + Dist * vz, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vz));
  Dist  = t1.DistanceToOut(pzero, vmz);
  valid = t1.Normal(pzero + Dist * vmz, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vmz));
  Dist  = t1.DistanceToOut(pzero, vxy);
  valid = t1.Normal(pzero + Dist * vxy, norm);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vxy));

  // SafetyToIn(P)
  Dist = t1.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));

  // DistanceToIn(P,V)
  Dist = t1.DistanceToIn(pbigx, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  // point is on (Rmin,z=-100) but travels away from the volume and will never get inside it
  Dist = test10.DistanceToIn(Vec_t(19.218716967888, 5.5354239324172, -100.0),
                             Vec_t(-0.25644483536346, -0.073799216676426, 0.96373737191901));
  std::cout << "D2I() mismatch: Line " << __LINE__ << ", test10::DistToIn =" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  Dist = test10.DistanceToOut(Vec_t(19.218716967888, 5.5354239324172, -100.0),
                              Vec_t(-0.25644483536346, -0.073799216676426, 0.96373737191901));
  // std::cout<<"D2O() mismatch: Line "<< __LINE__ <<", test10::DistToOut ="<<Dist<<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  // CalculateExtent
  Vec_t minExtent, maxExtent;
  t1.Extent(minExtent, maxExtent);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-50, -50, -50)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(50, 50, 50)));
  ctest10.Extent(minExtent, maxExtent);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-140, -140, -100)));

#ifndef VECGEOM_NO_SPECIALIZATION
  // Tests for Specialized Cones
  Precision sRmin1 = 0., sRmax1 = 5., sRmin2 = 0., sRmax2 = 5., sDz = 5., sSphi = 0., sDphi = vecgeom::kTwoPi;
  auto nonHollowTubeLikeCone =
      vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedCone>(sRmin1, sRmax1, sRmin2, sRmax2, sDz, sSphi, sDphi);
  sRmin1 = 2.;
  sRmin2 = 2.;
  auto hollowTubeLikeCone =
      vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedCone>(sRmin1, sRmax1, sRmin2, sRmax2, sDz, sSphi, sDphi);
  sDphi = vecgeom::kPi;
  auto hollowTubeLikeConeWithPiSector =
      vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedCone>(sRmin1, sRmax1, sRmin2, sRmax2, sDz, sSphi, sDphi);
  sDphi = vecgeom::kPi / 3.;
  auto hollowTubeLikeConeWithSmallerThanPiSector =
      vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedCone>(sRmin1, sRmax1, sRmin2, sRmax2, sDz, sSphi, sDphi);
  sDphi = 4 * vecgeom::kPi / 3.;
  auto hollowTubeLikeConeWithBiggerThanPiSector =
      vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedCone>(sRmin1, sRmax1, sRmin2, sRmax2, sDz, sSphi, sDphi);

  using NonHollowTubeLikeCone =
      vecgeom::SUnplacedImplAs<vecgeom::cxx::SUnplacedCone<vecgeom::cxx::ConeTypes::NonHollowCone>,
                               vecgeom::cxx::SUnplacedTube<vecgeom::cxx::TubeTypes::NonHollowTube>>;
  using HollowTubeLikeCone = vecgeom::SUnplacedImplAs<vecgeom::cxx::SUnplacedCone<vecgeom::cxx::ConeTypes::HollowCone>,
                                                      vecgeom::cxx::SUnplacedTube<vecgeom::cxx::TubeTypes::HollowTube>>;
  using HollowTubeLikeConeWithPiSector =
      vecgeom::SUnplacedImplAs<vecgeom::cxx::SUnplacedCone<vecgeom::cxx::ConeTypes::HollowConeWithPiSector>,
                               vecgeom::cxx::SUnplacedTube<vecgeom::cxx::TubeTypes::HollowTubeWithPiSector>>;
  using HollowTubeLikeConeWithSmallerThanPiSector =
      vecgeom::SUnplacedImplAs<vecgeom::cxx::SUnplacedCone<vecgeom::cxx::ConeTypes::HollowConeWithSmallerThanPiSector>,
                               vecgeom::cxx::SUnplacedTube<vecgeom::cxx::TubeTypes::HollowTubeWithSmallerThanPiSector>>;
  using HollowTubeLikeConeWithBiggerThanPiSector =
      vecgeom::SUnplacedImplAs<vecgeom::cxx::SUnplacedCone<vecgeom::cxx::ConeTypes::HollowConeWithBiggerThanPiSector>,
                               vecgeom::cxx::SUnplacedTube<vecgeom::cxx::TubeTypes::HollowTubeWithBiggerThanPiSector>>;

  // Checking type of TubeLikeCone, it should return true with pointer of TubeLikeCone
  VECGEOM_ASSERT(dynamic_cast<NonHollowTubeLikeCone *>(nonHollowTubeLikeCone));
  VECGEOM_ASSERT(dynamic_cast<HollowTubeLikeCone *>(hollowTubeLikeCone));
  VECGEOM_ASSERT(dynamic_cast<HollowTubeLikeConeWithPiSector *>(hollowTubeLikeConeWithPiSector));
  VECGEOM_ASSERT(dynamic_cast<HollowTubeLikeConeWithSmallerThanPiSector *>(hollowTubeLikeConeWithSmallerThanPiSector));
  VECGEOM_ASSERT(dynamic_cast<HollowTubeLikeConeWithBiggerThanPiSector *>(hollowTubeLikeConeWithBiggerThanPiSector));

#endif

  return true;
}

int main(int argc, char *argv[])
{
  TestCons<vecgeom::SimpleCone>();
  std::cout << "VecGeom Cone passed\n";
  return 0;
}
