//
// File:    TestTube.cpp
// Purpose: unit test for tube

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Tube.h"
#include "ApproxEqual.h"
#include "VecGeom/base/Global.h"

#include <cmath>

using namespace vecgeom;

bool testvecgeom = true;

const char *OutputInside(vecgeom::Inside_t side)
{
  using vecgeom::EnumInside;
  const char *insideChar;
  insideChar = (side == EnumInside::kInside ? "inside" : (side == EnumInside::kOutside ? "outside" : "surface"));
  return insideChar;
}

template <class Tube_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestTubs()
{
  std::cout.precision(16);
  vecgeom::EnumInside side;
  Vec_t pzero(0, 0, 0);
  Vec_t ptS(0, 0, 0);

  Precision kCarTolerance = vecgeom::kTolerance;
  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t ponxside(50, 0, 0);
  Vec_t ponyside(0, 50, 0);
  Vec_t ponzside(0, 0, 50);
  Vec_t ponlowcircle(50, 0, -50);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Precision Dist, vol, volCheck;
  Tube_t t1("Solid Tube #1", 0, 50, 50, 0, 2 * kPi);
  Tube_t t1a("Solid Tube #1", 0, 50, 50, 0, 0.5 * kPi);
  Tube_t t2("Hole Tube #2", 45, 50, 50, 0, 2 * kPi);
  Tube_t t2a("Hole Tube #2a", 5, 50, 50, 0, 2 * kPi);
  Tube_t t2b("Hole Tube #2b", 15, 50, 50, 0, 2 * kPi);
  Tube_t t2c("Hole Tube #2c", 25, 50, 50, 0, 2 * kPi);
  Tube_t t2d("Hole Tube #2d", 35, 50, 50, 0, 2 * kPi);
  Tube_t t3("Solid Sector #3", 0, 50, 50, 0.5 * kPi, 0.5 * kPi);
  Tube_t t4("Hole Sector #4", 45, 50, 50, 0.5 * kPi, 0.5 * kPi);
  Tube_t t5("Hole Sector #5", 50, 100, 50, 0.0, 1.5 * kPi);
  Tube_t t6("Solid Sector #3", 0, 50, 50, 0.5 * kPi, 1.5 * kPi);
  Tube_t tube6("tube6", 750, 760, 350, 0.31415926535897931, 5.6548667764616276);
  Tube_t tube7("tube7", 2200, 3200, 2500, -0.68977164349384879, 3.831364227270472);
  Tube_t tube8("tube8", 2550, 2580, 2000, 0, 2 * kPi);
  Tube_t tube9("tube9", 1150, 1180, 2000, 0, 2 * kPi);
  Tube_t tube10("tube10", 400, 405, 400, 0, 2 * kPi);
  Tube_t *clad = new Tube_t("clad", 90., 110., 105, 0., kPi); // external
  Tube_t *core = new Tube_t("core", 95., 105., 100, 0., kPi); // internal

  std::cout.precision(20);

  // Check name
  // assert(t1.GetName()=="Solid Tube #1");

  // Check cubic volume
  vol      = t1.Capacity();
  volCheck = 50 * 2 * kPi * 50 * 50;
  assert(ApproxEqual<Precision>(vol, volCheck));

  {
    // add a test for a previously fixed bug -- point near phi-surface of a wedged tube
    using vecgeom::cxx::kDegToRad;

    Tube_t jira174Tube("jira174Tube", 57.555599999999998, 205.55599999999998, 348, -5.5747247 * kDegToRad,
                       5.5747247 * kDegToRad);
    Vec_t pos174(92.4733, 5.32907e-15, 19.943);
    Vec_t dir174(0.303163, -0.492481, 0.815815);

    dir174 /= dir174.Mag();
    Precision din = jira174Tube.DistanceToIn(pos174, dir174);
    assert(ApproxEqual<Precision>(din, 0));
    Precision dout = jira174Tube.DistanceToOut(pos174, dir174); //, norm174, conv174);
    assert(ApproxEqual<Precision>(dout, 19.499));

    Vec_t dir206 = -dir174;
    pos174.y()   = -pos174.y();
    din          = jira174Tube.DistanceToIn(pos174, dir206);
    // std::cout<<"L"<<__LINE__<<": Dist=jira174tube: pos="<<pos174<<", dir="<< dir206 <<", DistToIn="<< din
    // <<std::endl;
    assert(ApproxEqual<Precision>(din, kInfLength));
    dout = jira174Tube.DistanceToOut(pos174, dir206); //, norm174, conv174);
    // std::cout<<"L"<<__LINE__<<": Dist=jira174tube: pos="<<pos174<<", dir="<< dir206 <<", DistToOut="<< dout
    // <<std::endl;
    assert(ApproxEqual<Precision>(dout, 0));
  }

  // Check Surface area
  vol      = t2.SurfaceArea();
  volCheck = 2. * kPi * (45 + 50) * (50 - 45 + 2 * 50);
  assert(ApproxEqual<Precision>(vol, volCheck));

  Tube_t myClad("myClad", 90.0, 110.0, 105.0, 0.0, kPi); // TEST MINE

  {
    // adding a test case found in VECGEOM-206: point on the Rmax surface, and going away - distanceToOut must be <=
    // zero
    Vec_t pos221(44.991, 21.816, 4.677);
    Vec_t dir221(0.16636, 0.64765, -0.74356);
    Dist = t1.DistanceToOut(pos221, dir221.Unit());
    assert(Dist < 0);
  }

  {
    // adding a test case found in VECGEOM-222: point on the outside, and going away - distanceToOut must be <= zero
    Vec_t pos222(60, -60, 10);
    Vec_t dir222(-1, 1, 0.001);
    Dist = t5.DistanceToIn(pos222, dir222.Unit());
    // std::cout<<" T5.DistToIn( "<< pos222 <<", "<< dir222 <<") = "<< Dist <<"\n";
    assert(ApproxEqual<Precision>(Dist, 134.8528474556));
  }

  // Check Inside
  assert(t1.Inside(pzero) == vecgeom::EInside::kInside);
  assert(t1.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(t1.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(t1.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(t1.Inside(ponzside) == vecgeom::EInside::kSurface);
  assert(t1a.Inside(pzero) == vecgeom::EInside::kSurface);
  assert(t1a.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(t1a.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(t1a.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(t1a.Inside(ponzside) == vecgeom::EInside::kSurface);
  assert(t2.Inside(pzero) == vecgeom::EInside::kOutside);
  assert(t2.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(t2.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(t2.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(t2.Inside(ponzside) == vecgeom::EInside::kOutside);
  assert(t2a.Inside(pzero) == vecgeom::EInside::kOutside);
  assert(t2a.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(t2a.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(t2a.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(t2a.Inside(ponzside) == vecgeom::EInside::kOutside);

  // Check Surface Normal
  Vec_t normal;
  bool valid;
  Vec_t norm;
  Precision p2 = 1. / std::sqrt(2.), p3 = 1. / std::sqrt(3.);
  valid = t1.Normal(ponxside, normal);
  assert(ApproxEqual(normal, vx));

  valid = t4.Normal(Vec_t(0., 50., 0.), normal);
  assert(ApproxEqual(normal, Vec_t(p2, p2, 0.)) && valid);
  valid = t4.Normal(Vec_t(0., 45., 0.), normal);
  assert(ApproxEqual(normal, Vec_t(p2, -p2, 0.)));
  valid = t4.Normal(Vec_t(0., 45., 50.), normal);
  assert(ApproxEqual(normal, Vec_t(p3, -p3, p3)));
  valid = t4.Normal(Vec_t(0., 45., -50.), normal);
  assert(ApproxEqual(normal, Vec_t(p3, -p3, -p3)));
  valid = t4.Normal(Vec_t(-50., 0., -50.), normal);
  assert(ApproxEqual(normal, Vec_t(-p3, -p3, -p3)));
  valid = t4.Normal(Vec_t(-50., 0., 0.), normal);
  assert(ApproxEqual(normal, Vec_t(-p2, -p2, 0.)));
  valid = t6.Normal(Vec_t(0., 0., 0.), normal);
  assert(ApproxEqual(normal, Vec_t(p2, p2, 0.)));

  // SafetyToOut(P)
  Dist = t1.SafetyToOut(pzero);
  assert(ApproxEqual<Precision>(Dist, 50));

  // DistanceToOut(P,V)
  Dist  = t1.DistanceToOut(pzero, vx);
  valid = t1.Normal(pzero + Dist * vx, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vx));

  Dist  = t1.DistanceToOut(pzero, vmx);
  valid = t1.Normal(pzero + Dist * vmx, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vmx));

  Dist  = t1.DistanceToOut(pzero, vy);
  valid = t1.Normal(pzero + Dist * vy, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vy));

  Dist  = t1.DistanceToOut(pzero, vmy);
  valid = t1.Normal(pzero + Dist * vmy, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vmy));

  Dist  = t1.DistanceToOut(pzero, vz);
  valid = t1.Normal(pzero + Dist * vz, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vz));

  Dist  = t1.DistanceToOut(pzero, vmz);
  valid = t1.Normal(pzero + Dist * vmz, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vmz));

  Dist  = t1.DistanceToOut(pzero, vxy);
  valid = t1.Normal(pzero + Dist * vxy, norm);
  assert(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(norm, vxy));

  Dist = t1.DistanceToOut(ponlowcircle + Vec_t(3.e-10, 0, 0), vmx + Vec_t(0, 0, -1.e-13));
  assert(ApproxEqual<Precision>(Dist, 100));

  Dist = t1.DistanceToOut(ponlowcircle, vz);
  assert(ApproxEqual<Precision>(Dist, 100));

  Dist = t2.DistanceToOut(pzero, vxy);
  //  std::cout<<"Dist=t2.DistanceToOut(pzero,vxy) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(ponxside, vmx);
  //  std::cout<<"Dist=t2.DistanceToOut(ponxside,vmx) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(ponxside, vmxmy);
  //  std::cout<<"Dist=t2.DistanceToOut(ponxside,vmxmy) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(ponxside, vz);
  //  std::cout<<"Dist=t2.DistanceToOut(ponxside,vz) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(pbigx, vx);
  //   std::cout<<"Dist=t2.DistanceToOut(pbigx,vx) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(pbigx, vxy);
  //   std::cout<<"Dist=t2.DistanceToOut(pbigx,vxy) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(pbigx, vz);
  //   std::cout<<"Dist=t2.DistanceToOut(pbigx,vz) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(Vec_t(45.5, 0, 0), vx);
  //  std::cout<<"Dist=t2.DistanceToOut((45.5,0,0),vx) = "<<Dist<<std::endl;

  Dist = t2.DistanceToOut(Vec_t(49.5, 0, 0), vx);
  //  std::cout<<"Dist=t2.DistanceToOut((49.5,0,0),vx) = "<<Dist<<std::endl;

  Dist = t3.DistanceToOut(Vec_t(0, 10, 0), vx);
  // std::cout<<"Dist=t3.DistanceToOut((0,10,0),vx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 0));

  Dist = t3.DistanceToOut(Vec_t(0.5, 10, 0), vx); // checking an outside point
  assert(t3.Inside(Vec_t(0.5, 10, 0)) == vecgeom::EInside::kOutside);
  // std::cout<<"Dist=t3.DistanceToOut((0.5,10,0),vx) = "<<Dist<<std::endl;
  // assert(ApproxEqual<Precision>(Dist, 48.489795)); // distance  can't be positive, buggy test case
  assert(Dist < 0.);

  Dist = t3.DistanceToOut(Vec_t(-0.5, 9, 0), vx);
  // std::cout<<"Dist=t3.DistanceToOut((-0.5,9,0),vx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 0.5));

  Dist = t3.DistanceToOut(Vec_t(-5, 9.5, 0), vx);
  // std::cout<<"Dist=t3.DistanceToOut((-5,9.5,0),vx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 5));

  Dist = t3.DistanceToOut(Vec_t(-5, 9.5, 0), vmy);
  // std::cout<<"Dist=t3.DistanceToOut((-5,9.5,0),vmy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 9.5));

  Dist = t3.DistanceToOut(Vec_t(-5, 9, 0), vxmy);
  // std::cout<<"Dist=t3.DistanceToOut((-5,9,0),vxmy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 7.0710678));

  // SafetyToIn(P)

  Dist = t1.SafetyToIn(pbigx);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigmx);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigy);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigmy);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigz);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.SafetyToIn(pbigmz);
  assert(ApproxEqual<Precision>(Dist, 50));

  // DistanceToIn(P,V)

  Dist = t1.DistanceToIn(pbigx, vmx);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmx, vx);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigy, vmy);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmy, vy);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigz, vmz);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmz, vz);
  assert(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigx, vxy);
  assert(ApproxEqual<Precision>(Dist, kInfLength));

  Dist = t1a.DistanceToIn(pbigz, vmz);
  assert(ApproxEqual<Precision>(Dist, 50));

  Dist = t2.DistanceToIn(Vec_t(45.5, 0, 0), vx);
  //  std::cout<<"Dist=t2.DistanceToIn((45.5,0,0),vx) = "<<Dist<<std::endl;

  Dist = t2.DistanceToIn(Vec_t(45.5, 0, 0), vmx);
  //  std::cout<<"Dist=t2.DistanceToIn((45.5,0,0),vmx) = "<<Dist<<std::endl;

  Dist = t2.DistanceToIn(Vec_t(49.5, 0, 0), vmx);
  //  std::cout<<"Dist=t2.DistanceToIn((49.5,0,0),vmx) = "<<Dist<<std::endl;

  Dist = t2.DistanceToIn(Vec_t(49.5, 0, 0), vx);
  //   std::cout<<"Dist=t2.DistanceToIn((49.5,0,0),vx) = "<<Dist<<std::endl;

  Dist = t3.DistanceToIn(Vec_t(49.5, 0, 0), vmx);
  //  std::cout<<"Dist=t2.DistanceToIn((49.5,0,0),vmx) = "<<Dist<<std::endl;

  Dist = t3.DistanceToIn(Vec_t(49.5, 5, 0), vmx);
  //  std::cout<<"Dist=t2.DistanceToIn((49.5,5,0),vmx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 49.5));

  Dist = t3.DistanceToIn(Vec_t(49.5, -0.5, 0), vmx);
  //  std::cout<<"Dist=t2.DistanceToIn((49.5,-0.5,0),vmx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, kInfLength));

  Dist = t5.DistanceToIn(Vec_t(30.0, -20.0, 0), vxy);
  // std::cout<<"Dist=t5.DistanceToIn((30.0,-20.0,0),vxy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 28.284271));

  // This ray is passing through an edge, may or may not hit depending on rounding
  Dist = t5.DistanceToIn(Vec_t(30.0, -70.0, 0), vxy);
  // std::cout<<"Dist=t5.DistanceToIn((30.0,-70.0,0),vxy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, kInfLength) || ApproxEqual<Precision>(Dist, 70. * std::sqrt(2.)));

  Dist = t5.DistanceToIn(Vec_t(30.0, -20.0, 0), vmxmy);
  //  std::cout<<"Dist=t5.DistanceToIn((30.0,-20.0,0),vmxmy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 42.426407));

  // This ray is passing through an edge, may or may not hit depending on rounding
  Dist = t5.DistanceToIn(Vec_t(30.0, -70.0, 0), vmxmy);
  // std::cout<<"Dist=t5.DistanceToIn((30.0,-70.0,0),vmxmy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, kInfLength) || ApproxEqual<Precision>(Dist, 30. * std::sqrt(2.)));

  Dist = t5.DistanceToIn(Vec_t(50.0, -20.0, 0), vy);
  // std::cout<<"Dist=t5.DistanceToIn((50.0,-20.0,0),vy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 20));

  // This ray is passing through an edge, may or may not hit depending on rounding
  Dist = t5.DistanceToIn(Vec_t(100.0, -20.0, 0), vy);
  // std::cout<<"Dist=t5.DistanceToIn((100.0,-20.0,0),vy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, kInfLength) || ApproxEqual<Precision>(Dist, 20.));

  Dist = t5.DistanceToIn(Vec_t(30.0, -50.0, 0), vmx);
  //  std::cout<<"Dist=t5.DistanceToIn((30.0,-50.0,0),vmx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 30));

  // This ray is passing through an edge, may or may not hit depending on rounding
  Dist = t5.DistanceToIn(Vec_t(30.0, -100.0, 0), vmx);
  //  std::cout<<"Dist=t5.DistanceToIn((30.0,-100.0,0),vmx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, kInfLength) || ApproxEqual<Precision>(Dist, 30.));

  // ********************************

  // Tubs from Problem reports

  // Make a tub

  Tube_t *arc = new Tube_t("outer", 1000, 1100, 10, -kPi / 12., kPi / 6.);

  // First issue:
  //   A point on the start phi surface just beyond the
  //   start angle but still well within tolerance
  //   is found to be "outside" by Tube_t::Inside
  //
  //   pt1 = exactly on phi surface (within precision)
  //   pt2 = t1 but slightly higher, and still on tolerant surface
  //   pt3 = t1 but slightly lower, and still on tolerant surface
  //

  Vec_t pt1(1050 * std::cos(-kPi / 12.), 1050 * std::sin(-kPi / 12.), 0);

  Vec_t pt2 = pt1 + Vec_t(0, 0.001 * kCarTolerance, 0);
  Vec_t pt3 = pt1 - Vec_t(0, 0.001 * kCarTolerance, 0);

  vecgeom::EnumInside a1 = arc->Inside(pt1);
  vecgeom::EnumInside a2 = arc->Inside(pt2);
  vecgeom::EnumInside a3 = arc->Inside(pt3);

  // std::cout << "Point pt1 is " << OutputInside(a1) << std::endl;
  assert(a1 == vecgeom::EInside::kSurface);
  // std::cout << "Point pt2 is " << OutputInside(a2) << std::endl;
  assert(a2 == vecgeom::EInside::kSurface);
  // std::cout << "Point pt3 is " << OutputInside(a3) << std::endl;
  assert(a3 == vecgeom::EInside::kSurface);

  assert(t1.Inside(pzero) == vecgeom::EInside::kInside);
  assert(t1.Inside(pbigx) == vecgeom::EInside::kOutside);

  vecgeom::EnumInside in = t5.Inside(Vec_t(60, -0.001 * kCarTolerance, 0));
  assert(in == vecgeom::EInside::kSurface);
  //    std::cout<<"t5.Inside(Vec_t(60,-0.001*kCarTolerance,0)) = "
  //     <<OutputInside(in)<<std::endl;
  in = tube10.Inside(Vec_t(-114.8213313833317, 382.7843220719649, -32.20788536438663));
  assert(in == vecgeom::EInside::kOutside);
  // std::cout<<"tube10.Inside(Vec_t(-114.821...)) = "<<OutputInside(in)<<std::endl;

  // bug #76
  Dist = tube6.DistanceToOut(Vec_t(-388.20504321896431, -641.71398957741451, 332.85995254027955),
                             Vec_t(-0.47312863350457468, -0.782046391443315, 0.40565100491504164));
  // std::cout<<"Dist=tube6.DistanceToOut(p,v) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 10.940583));

  // bug #91
  Dist = tube7.DistanceToOut(Vec_t(-2460, 1030, -2500),
                             Vec_t(-0.086580540180167642, 0.070084247882560638, 0.9937766390194761));
  assert(ApproxEqual<Precision>(Dist, 4950.348576972614));

  Dist = tube8.DistanceToOut(Vec_t(6.71645645882942, 2579.415860329989, -1.519530725281157),
                             Vec_t(-0.6305220496340839, -0.07780451841562354, 0.7722618738739774));
  assert(ApproxEqual<Precision>(Dist, 1022.64931421));

  Dist = tube9.DistanceToOut(Vec_t(2.267347771505638, 1170.164934028592, 4.820317321984064),
                             Vec_t(-0.1443054266272111, -0.01508874701037938, 0.9894181489944458));
  assert(ApproxEqual<Precision>(Dist, 2016.51817758));

  Dist = t1a.DistanceToOut(Vec_t(0., 0., 50.), vx);
  // std::cout<<"Dist=t1a.DistanceToOut((0,0,50),vx) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 50));

  Dist = t1a.DistanceToOut(Vec_t(0., 5., 50.), vmy);
  // std::cout<<"Mismatch: L"<<__LINE__<<": Dist=t1a.DistanceToOut((0,5,50),vmy) = "<<Dist<<std::endl;
  assert(ApproxEqual<Precision>(Dist, 5));

  // std::cout<<std::endl ;

  // Bug 810

  Vec_t pTmp(0., 0., 0.);

  Dist = clad->DistanceToIn(pTmp, vy);
  pTmp += Dist * vy;
  // std::cout<<"Dist="<< Dist <<" --> pTmp = "<< pTmp << std::endl;
  side = core->Inside(pTmp);
  assert(side == vecgeom::EInside::kOutside);
  // std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
  side = clad->Inside(pTmp);
  // std::cout<<"clad->Inside(pTmp) = "<< OutputInside(side) <<std::endl;
  assert(side == vecgeom::EInside::kSurface);

  Dist = core->DistanceToIn(pTmp, vy);
  pTmp += Dist * vy;
  // std::cout<<"Dist="<< Dist <<" --> pTmp = "<< pTmp <<"\n";
  side = core->Inside(pTmp);
  assert(side == vecgeom::EInside::kSurface);
  // std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
  side = clad->Inside(pTmp);
  assert(side == vecgeom::EInside::kInside);
  // std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
  Dist = core->DistanceToOut(pTmp, vy);
  pTmp += Dist * vy;
  // std::cout<<"pTmpX = "<<pTmp.x<<";  pTmpY = "<<pTmp.y<<";  pTmpZ = "<<pTmp.z<<std::endl;
  side = core->Inside(pTmp);
  assert(side == vecgeom::EInside::kSurface);
  // std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
  side = clad->Inside(pTmp);
  assert(side == vecgeom::EInside::kInside);
  // std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;

  Dist = clad->DistanceToOut(pTmp, vy);
  pTmp += Dist * vy;
  // std::cout<<"pTmpX = "<<pTmp.x<<";  pTmpY = "<<pTmp.y<<";  pTmpZ = "<<pTmp.z<<std::endl;
  side = core->Inside(pTmp);
  assert(side == vecgeom::EInside::kOutside);
  // std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
  side = clad->Inside(pTmp);
  assert(side == vecgeom::EInside::kSurface);
  // std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;

  Vec_t pSN1 = Vec_t(33.315052227388207, 37.284142675357259, 33.366096020078537);
  Tube_t t4SN("Hole Sector #4", 45, 50, 50, kPi / 4., kPi / 8.);

  in = t4SN.Inside(pSN1);
  assert(in == vecgeom::EInside::kSurface);
  valid = t4SN.Normal(pSN1, normal);

  // Check Extent and cached BBox
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  t1.Extent(minExtent, maxExtent);
  t1.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
  assert(ApproxEqual(minExtent, Vec_t(-50, -50, -50)));
  assert(ApproxEqual(maxExtent, Vec_t(50, 50, 50)));
  assert(ApproxEqual(minExtent, minBBox));
  assert(ApproxEqual(maxExtent, maxBBox));
  t2.Extent(minExtent, maxExtent);
  t2.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
  assert(ApproxEqual(minExtent, Vec_t(-50, -50, -50)));
  assert(ApproxEqual(maxExtent, Vec_t(50, 50, 50)));
  assert(ApproxEqual(minExtent, minBBox));
  assert(ApproxEqual(maxExtent, maxBBox));

  /* ********************************
  ************************************ */

  // point on boundary of BeamTube1 in CMS
  // track exiting and DO has to be 0
  Tube_t BT1("Solid Tube #1", 0.0, 2.25, 72.475, 0.0, 2.0 * kPi);
  //    Dist = BT1.DistanceToOut(Vec_t (1.90682136437479,
  //                                    1.1943752694877201,
  //                                    -41.601888140951587),
  //                             (Vec_t (0.060059263937965943,
  //                                    0.037619307655839145,
  //                                    0.97241220828092478)).Normalized()
  //
  //                                    ,
  //
  //                                    ,
  //
  //    );
  //   assert( ApproxEqual<Precision>(Dist,0.) && " DO not larger than 0 ");

  Tube_t testTube("testTube", 0., 5., 5., 0., 2 * kPi);
  Vec_t pOutZ(2., 0., 6.);
  normal.Set(0., 0., 0.);
  valid = testTube.Normal(pOutZ, normal);
  assert(ApproxEqual(normal, Vec_t(0., 0., 1.)));
  // std::cout<<"Normal for Point Outside +Z : "<< normal << std::endl;
  normal.Set(0., 0., 0.);
  Vec_t pOutX(6., 0., 0.);
  valid = testTube.Normal(pOutX, normal);
  assert(ApproxEqual(normal, Vec_t(1., 0., 0.)));
  // std::cout<<"Normal for Point Outside +X : "<< normal << std::endl;

  Tube_t testTube2("testTube", 3., 5., 5., 0., 2 * kPi);
  normal.Set(0., 0., 0.);
  valid = testTube2.Normal(pOutZ, normal);
  // std::cout<<"Normal for Point Outside +Z : "<< normal << std::endl;
  Vec_t pOutXUp(6., 0., 4.);
  normal.Set(0., 0., 0.);
  valid = testTube2.Normal(pOutXUp, normal);
  assert(ApproxEqual(normal, Vec_t(1., 0., 0.)));
  // std::cout<<"Normal for Point Outside +XUp : "<< normal << std::endl;
  Vec_t pOutXin(2., 0., 4.);
  normal.Set(0., 0., 0.);
  valid = testTube2.Normal(pOutXin, normal);
  assert(ApproxEqual(normal, Vec_t(-1., 0., 0.)));
  // std::cout<<"Normal for Point Outside +OutXin : "<< normal << std::endl;
  normal.Set(0., 0., 0.);
  Vec_t pOutXOutZ(6., 0., 6.);
  valid = testTube2.Normal(pOutXOutZ, normal);
  // std::cout<<"Normal for Point Outside pOutXOutZ : "<< normal << std::endl;
  Vec_t pOutXOutYOutZ(6., 6., 6.);
  normal.Set(0., 0., 0.);
  valid = testTube2.Normal(pOutXOutYOutZ, normal);
  // std::cout<<"Normal for Point Outside pOutXOutYOutZ : "<< normal << std::endl;

  Vec_t pOutXOutYInZ(2., 2., 4.);
  normal.Set(0., 0., 0.);
  valid = testTube2.Normal(pOutXOutYInZ, normal);
  assert(ApproxEqual(normal, Vec_t(-1 / std::sqrt(2.), -1. / std::sqrt(2.), 0.)));
  // std::cout<<"Normal for Point Outside pOutXOutYInZ : "<< normal << std::endl;

  // Added Normal tests pointed by Evgueni Tcherniaev in jira-issue-443
  normal.Set(0., 0., 0.);
  Tube_t tubeN("SolidTube", 0., 200., 200., 0., vecgeom::kTwoPi);
  Vec_t ptN1(199.99999999, 0, 190);
  valid = tubeN.Normal(ptN1, normal);
  assert(ApproxEqual(normal, Vec_t(1., 0., 0.)));

  normal.Set(0., 0., 0.);
  Vec_t ptN2(100, 0, 199.99999999);
  valid = tubeN.Normal(ptN2, normal);
  assert(ApproxEqual(normal, Vec_t(0., 0., 1.)));

  //.. Added Some more Normal test for the points on the circular edges of tube
  //   These corresponds to the test cases pointed by Evgueni Tcherniaev in JIRA issue VECGEOM-439
  Precision rmin = 100., rmax = 200., dz = 200.;
  Tube_t hollowTube("testHolloTube", rmin, rmax, dz, 0, 2 * kPi);
  Precision rad = 0.;
  for (int j = 0; j < 2; j++) {
    if (j == 0) // inspecting point on inner radius
      rad = rmin;
    else // inspecting point on outer radius
      rad = rmax;

    // For Top Z
    for (int i = 0; i <= 360; i++) {

      Vec_t pt(rad * std::cos(i * vecgeom::kDegToRad), rad * std::sin(i * vecgeom::kDegToRad), dz);
      Vec_t normal(0., 0., 0.);
      hollowTube.Normal(pt, normal);
      assert(normal.z() != 0. && normal.z() > 0.);
    }
    // For Bottom Z
    for (int i = 0; i <= 360; i++) {

      Vec_t pt(rad * std::cos(i * vecgeom::kDegToRad), rad * std::sin(i * vecgeom::kDegToRad), -dz);
      Vec_t normal(0., 0., 0.);
      hollowTube.Normal(pt, normal);
      assert(normal.z() != 0. && normal.z() < 0.);
    }
  }
  return true;
}

int main(int argc, char *argv[])
{
  TestTubs<vecgeom::SimpleTube>();
  std::cout << "VecGeom tube passed\n";

  return 0;
}
