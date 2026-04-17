//
//
// TestTorus

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "ApproxEqual.h"
#include "VecGeom/volumes/Torus2.h"

#include <cmath>
using vecgeom::kPi;
using vecgeom::Precision;

template <class Torus_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool testTorus()
{
  int i;
  Precision Rtor = 100;
  Precision Rmax = Rtor * 0.9;
  Precision Rmin = Rtor * 0.1;
  Precision x;
  Precision z;

  Precision Dist, dist, vol, volCheck;
  Vec_t normal;
  bool valid;

  Precision tolerance = 1e-9;

  Vec_t pzero(0, 0, 0);

  Vec_t pbigx(240, 0, 0), pbigy(0, 240, 0), pbigz(0, 0, 240);
  Vec_t pbigmx(-240, 0, 0), pbigmy(0, -240, 0), pbigmz(0, 0, -240);

  Vec_t ponrmax(190, 0, 0);
  Vec_t ponrmin(0, 110, 0);
  Vec_t ponrtor(0, 100, 0);
  Vec_t ponphi1(100 / std::sqrt(2.), 100 / std::sqrt(2.), 0);
  Vec_t ponphi2(-100 / std::sqrt(2.), 100 / std::sqrt(2.), 0);
  Vec_t ponphi12(190 / std::sqrt(2.), 190 / std::sqrt(2.), 0);
  Vec_t ponphi22(-120 / std::sqrt(2.), 120 / std::sqrt(2.), 0);
  Vec_t ponphi23(-120 / std::sqrt(2.) + 0.5, 120 / std::sqrt(2.), 0);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Vec_t pstart((Rtor + Rmax) / std::sqrt(2.0), (Rtor + Rmax) / std::sqrt(2.0), 0);
  Vec_t vdirect(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Vec_t pother(90, 0, 0);
  vdirect = vdirect.Unit();
  Vec_t p1;
  Vec_t v1(1, 0, 0);
  v1 = v1.Unit();

  std::cout << "Starting Torus unit test..." << std::endl;
  // Check torus roots

  Torus_t t1("Solid Torus #1", 0, Rmax, Rtor, 0, vecgeom::kTwoPi);
  Torus_t t2("Hole cutted Torus #2", Rmin, Rmax, Rtor, 0, kPi / 2.); // kPi/4., kPi/2.);
  Torus_t tn2("tn2", Rmin, Rmax, Rtor, vecgeom::kPi / 2., vecgeom::kPi / 2.);
  Torus_t tn3("tn3", Rmin, Rmax, Rtor, vecgeom::kPi / 2., 3 * vecgeom::kPi / 2.);
  Torus_t t3("Hole cutted Torus #3", 4 * Rmin, Rmax, Rtor, vecgeom::kPi / 2. - vecgeom::kPi / 24, vecgeom::kPi / 12);
  Torus_t t4("Solid Torus #4", 0, Rtor - 2.e3 * tolerance, Rtor, 0, vecgeom::kTwoPi);
  Torus_t t5("Solid cutted Torus #5", 0, Rtor - 2.e3 * tolerance, Rtor, vecgeom::kPi / 4, vecgeom::kPi / 2);
  Torus_t *aTub = new Torus_t("Ring1", 0, 100, 1000, 0, vecgeom::kTwoPi);
  Torus_t t6("t6", 100, 150, 200, 0, vecgeom::kPi / 3);
  Torus_t *clad = new Torus_t("clad", 0., 10., 100., 0., vecgeom::kPi); // external
  Torus_t *core = new Torus_t("core", 0., 5, 100, 0., vecgeom::kPi);    // internal

  Torus_t *cmsEECool3 = new Torus_t("cmsEECool3", 0, 6, 1640, 0.09599, 1.4312);
  Vec_t temp1         = Vec_t(-0.646752, -0.762700, -0.000012);
  temp1               = temp1 / (temp1.Mag());

  // std::cout << "Dout=" << cmsEECool3->DistanceToOut(Vec_t(1230.993171, 1078.075896, -2.947036), temp1)
  //          << std::endl; //: inf / inf / inf / 1.57856
  // std::cout<<"In="<<cmsEECool3->Inside(Vec_t(1230.993171, 1078.075896, -2.947036))<<"
  // Rtor="<<std::sqrt(1230.99*1230.99+1078.07*1078.07)<<std::endl;
  VECGEOM_ASSERT(cmsEECool3->Inside(Vec_t(1230.993171, 1078.075896, -2.947036)) == vecgeom::EInside::kInside);

  Vec_t p1t6(60.73813233071262, -27.28494547459707, 37.47827539879173);

  Vec_t vt6(0.3059312222729116, 0.8329513862588347, -0.461083588265824);

  Vec_t p2t6(70.75950555416668, -3.552713678800501e-15, 22.37458414788935);

  // Check cubic volume

  vol      = t1.Capacity();
  volCheck = vecgeom::kTwoPi * vecgeom::kPi * Rtor * (Rmax * Rmax);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  vol      = t2.Capacity();
  volCheck = vecgeom::kPi / 2. * vecgeom::kPi * Rtor * (Rmax * Rmax - Rmin * Rmin);

  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check Inside
  // std::cout<<t1.Inside(pzero)<<std::endl;
  VECGEOM_ASSERT(t1.Inside(pzero) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(t1.Inside(pbigx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(t1.Inside(ponrmax) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(t2.Inside(ponrmin) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(t2.Inside(pbigx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(t2.Inside(pbigy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(t2.Inside(ponphi1) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(t2.Inside(ponphi2) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(t2.Inside(Vec_t(20, 0, 0)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(t2.Inside(Vec_t(0, 20, 0)) == vecgeom::EInside::kSurface);

  vecgeom::EnumInside side;
  side = t6.Inside(p1t6);
  // std::cout << "t6.Inside(p1t6) = " << side << std::endl;
  side = t6.Inside(p2t6);
  // std::cout << "t6.Inside(p2t6) = " << side << std::endl;
  VECGEOM_ASSERT(t6.Inside(p2t6) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(side == vecgeom::EInside::kSurface);
  // Check Surface Normal

  Precision p2 = 1. / std::sqrt(2.); // ,p3=1./std::sqrt(3.);

  valid = t1.Normal(ponrmax, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, vx) && valid);
  valid = t1.Normal(Vec_t(0., 190., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, vy));
  valid = tn2.Normal(Vec_t(0., Rtor + Rmax, 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, p2, 0.)));
  valid = tn2.Normal(Vec_t(0., Rtor + Rmin, 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, -p2, 0.)));
  valid = tn2.Normal(Vec_t(0., Rtor - Rmin, 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, p2, 0.)));
  valid = tn2.Normal(Vec_t(0., Rtor - Rmax, 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, -p2, 0.)));
  valid = tn3.Normal(Vec_t(Rtor, 0., Rmax), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., p2, p2)));
  valid = tn3.Normal(Vec_t(0., Rtor, Rmax), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, 0., p2)));

  valid = t2.Normal(ponrmin, normal);
  // std::cout<<" normal="<<normal<<std::endl;
  VECGEOM_ASSERT(ApproxEqual(normal, vmxmy));
  valid = t2.Normal(Vec_t(20., 0., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, vmy));
  valid = t2.Normal(Vec_t(0, 20, 0), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, vmx));

  // SafetyToOut(P)
  Dist = t1.SafetyToOut(ponrmin);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = t1.SafetyToOut(ponrmax);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  // later: why it was introduced, while they are outside (see above)
  Dist = t2.SafetyToOut(Vec_t(20, 0, 0));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = t2.SafetyToOut(Vec_t(0, 20, 0));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  // DistanceToOut(P,V)
  Dist = t1.DistanceToOut(ponrmax, vx);
  // std::cout << "t1.DistanceToOut(p,vx...) = " << Dist << " norm=" << normal << std::endl;
  valid = t1.Normal(ponrmax + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vx));

  Vec_t ptest = Vec_t(130, 0, 0);
  Dist        = t1.DistanceToOut(ptest, vx);
  // Dist=t1.DistanceToOut(Vec_t(130,0.00001,0.00001),Vec_t(1,0.001,0.001).Unit(),normal,convex);
  // std::cout << "t1.DistanceToOut(ponphi1,vz,...) = " << Dist << " n=" << normal << std::endl;
  valid = t1.Normal(ptest + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60) && ApproxEqual(normal, vx));

  Dist  = t1.DistanceToOut(ponrmin, vy);
  valid = t1.Normal(ponrmin + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80) && ApproxEqual(normal, vy));
  Dist  = t1.DistanceToOut(ponrmin, vmy);
  valid = t1.Normal(ponrmin + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100));

  // std::cout << "(vz + Vec_t(0.00001, 0.000001, 0): " << (vz + Vec_t(0.00001, 0.000001, 0)) << '\n';
  // std::cout << "Vec_t(0.00001, 0.000001, 0)).Unit() = " << (vz + Vec_t(0.00001, 0.000001, 0)).Unit() << '\n';
  Dist = t1.DistanceToOut(Vec_t(100, 0, 0), (vz + Vec_t(0.00001, 0.000001, 0)).Unit());
  // std::cout << "t1.DistanceToOut(100,0,0,vz,...) = " << Dist << " n=" << norm << std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 90));
  // std::cout << "Dist=t2.DistanceToOut(ponphi12,vy) = " << Dist << ", n=" << norm << ", -vy = " << -vy << std::endl;
  // std::cout << "norm: " << norm << '\n';
  Dist = t2.DistanceToOut(Vec_t(7.07106781186547524400844362, 7.07106781186547524400844362, 0), vmx);
  // std::cout << "Dist = t2.DistanceToOut(Vec_t(7.07106781186547524400844362, 7.07106781186547524400844362, 0), vmx,
  // normal, convex) = " << Dist << std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  Vec_t test(0., 0., 1);
  for (int i = 1; i < 5; i++) {
    Dist = t1.DistanceToOut(Vec_t(90, 0, 0), test);
    // std::cout << " i=" << i << " Dist=t2.DistanceToOut(90,test) = " << Dist << " n=" << norm << std::endl;
    test = test + Vec_t(0.00001, 0.00001, 0);
    test = test.Unit();
  }

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
  //    std::cout<<"Dist=t1.SafetyToIn (pbigz) = "<<Dist<<std::endl;
  //    VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,50));
  Dist = t1.SafetyToIn(pbigmz);
  //    std::cout<<"Dist=t1.SafetyToIn (pbigmz) = "<<Dist<<std::endl;
  //    VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,50));

  // DistanceToIn(P,V)
  // std::cout << "LOGT pbigx: " << pbigx << '\n';
  // std::cout << "LOGT vmx: " << vmx << '\n';
  Dist = t1.DistanceToIn(pbigx, vmx);
  // std::cout << "LOGT Dist = t1.DistanceToIn(pbigx, vmx): " << Dist << '\n';
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = t1.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));
  Dist = t1.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));
  Dist = t1.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));
  Dist = t1.DistanceToIn(ponrmax, vx);
  // std::cout << "Dist=t1.DIN (p,v) = " << Dist << std::endl;
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,vecgeom::kInfLength));
  Dist = t1.DistanceToIn(ponrmax, vmx);
  // std::cout << "Dist=t1.DIN (p,v) = " << Dist << std::endl;
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,0));

  // Vec_t vnew(1,0,0) ;
  // vnew.rotateZ(pi/4-5*1e-9) ;    // old test: check pzero with vxy
  // Dist=t2.DistanceToIn(pzero,vnew);
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,vecgeom::kInfLength));
  // std::cout << "Dist=t2.DIN (p,v) = " << Dist << std::endl;
  Dist = t2.DistanceToIn(pzero, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 10));
  Dist = t2.DistanceToIn(ponphi12, vy);
  // std::cout << "Dist = t2.DistanceToIn(ponphi12, vy) = " << Dist << ", ponphi12 = " << ponphi12 << ", vy = " << vy <<
  // std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));
  Dist = t2.DistanceToIn(ponphi12, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = t2.DistanceToIn(ponphi1, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 13.550819613108856743)); // Not sure about this
  // Torus t2 is ends at pi/2 rad, we expect infinity
  Dist = t2.DistanceToIn(ponrmin, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));
  Dist = t2.DistanceToIn(ponrmin, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));

  Dist = t3.DistanceToIn(ponrtor, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40));
  Dist = t3.DistanceToIn(ponrtor, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40));
  Dist = t3.DistanceToIn(ponrtor, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40));
  Dist = t3.DistanceToIn(ponrtor, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40));
  Dist = t3.DistanceToIn(ponrtor, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));
  Dist = t3.DistanceToIn(ponrtor, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, vecgeom::kInfLength));

  Dist = t6.DistanceToIn(p1t6, vt6);
  // std::cout<<"t6.DistanceToIn(p1t6,vt6) = "<<Dist<<std::endl;

  // Bug 810

  Vec_t pTmp(0., 0., 0.);

  dist = clad->DistanceToIn(pTmp, vy);
  pTmp += dist * vy;
  // std::cout << "pTmpX = " << pTmp.x() << ";  pTmpY = " << pTmp.y() << ";  pTmpZ = " << pTmp.z() << std::endl;
  side = core->Inside(pTmp);
  // std::cout << "core->Inside(pTmp) = " << side << std::endl;
  side = clad->Inside(pTmp);
  // std::cout << "clad->Inside(pTmp) = " << side << std::endl;

  dist = core->DistanceToIn(pTmp, vy);
  pTmp += dist * vy;
  // std::cout << "pTmpX = " << pTmp.x() << ";  pTmpY = " << pTmp.y() << ";  pTmpZ = " << pTmp.z() << std::endl;
  side = core->Inside(pTmp);
  // std::cout << "core->Inside(pTmp) = " << side << std::endl;
  side = clad->Inside(pTmp);
  // std::cout << "clad->Inside(pTmp) = " << side << std::endl;

  dist = core->DistanceToOut(pTmp, vy);
  pTmp += dist * vy;
  // std::cout << "pTmpX = " << pTmp.x() << ";  pTmpY = " << pTmp.y() << ";  pTmpZ = " << pTmp.z() << std::endl;
  side = core->Inside(pTmp);
  // std::cout << "core->Inside(pTmp) = " << side << std::endl;
  side = clad->Inside(pTmp);
  // std::cout << "clad->Inside(pTmp) = " << side << std::endl;

  dist = clad->DistanceToOut(pTmp, vy);
  pTmp += dist * vy;
  // std::cout << "pTmpX = " << pTmp.x() << ";  pTmpY = " << pTmp.y() << ";  pTmpZ = " << pTmp.z() << std::endl;
  side = core->Inside(pTmp);
  // std::cout << "core->Inside(pTmp) = " << side << std::endl;
  side = clad->Inside(pTmp);
  // std::cout << "clad->Inside(pTmp) = " << side << std::endl;

  // Check for Distance to In ( start from an external point )

  for (i = 0; i < 12; i++) {
    x  = -1200;
    z  = Precision(i) / 10;
    p1 = Vec_t(x, 0, z);
    //     std::cout << p1 << " - " << v1 << std::endl;

    Dist = aTub->DistanceToIn(p1, v1);
    //     std::cout << "Distance to in dir: " << Dist ;

    Dist = aTub->DistanceToOut(p1, v1);
    //     std::cout << "   Distance to out dir: " << Dist << std::endl ;

    // std::cout << "Distance to in : " << aTub->DistanceToIn (p1);
    //  std::cout << "   Distance to out : " << aTub->DistanceToOut (p1)
    //    << std::endl;
    //     std::cout << "   Inside : " << aTub->Inside (p1);
    //     std::cout << std::endl;
  }

  // CalculateExtent
  std::cout << "Test passed." << std::endl;

  return true;
}

int main()
{
#ifdef NDEBUG
  G4Exception("FAIL: *** Assertions must be compiled in! ***");
#endif
  VECGEOM_ASSERT(testTorus<vecgeom::SimpleTorus2>());
  return 0;
}
