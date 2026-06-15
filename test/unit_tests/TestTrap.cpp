//
// File:    TestTrap.cpp
// Purpose: Unit tests for the trapezoid
//

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "VecCore/VecMath.h"
#include "VecGeom/volumes/Box.h"
#include "ApproxEqual.h"
#include "VecGeom/volumes/Trapezoid.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"
#include <cmath>
#include "VecGeom/management/GeoManager.h"

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

using vecgeom::kInfLength;
using vecgeom::kPi;
using vecgeom::Precision;
bool testvecgeom = true;

const Precision deg = atan(1.0) / 45.;

template <class Trap_t>
bool TestTrap()
{
  using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
  Vec_t pzero(0, 0, 0);
  Vec_t ponxside(20, 0, 0), ponyside(0, 30, 0), ponzside(0, 0, 40);
  Vec_t ponmxside(-20, 0, 0), ponmyside(0, -30, 0), ponmzside(0, 0, -40);
  Vec_t ponzsidey(0, 25, 40), ponmzsidey(0, 20, -40);

  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100), pbig(100, 100, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Vec_t vxmz(1 / std::sqrt(2.0), 0, -1 / std::sqrt(2.0));
  Vec_t vymz(0, 1 / std::sqrt(2.0), -1 / std::sqrt(2.0));
  Vec_t vmxmz(-1 / std::sqrt(2.0), 0, -1 / std::sqrt(2.0));
  Vec_t vmymz(0, -1 / std::sqrt(2.0), -1 / std::sqrt(2.0));
  Vec_t vxz(1 / std::sqrt(2.0), 0, 1 / std::sqrt(2.0));
  Vec_t vyz(0, 1 / std::sqrt(2.0), 1 / std::sqrt(2.0));

  Precision Dist, dist, vol, volCheck;
  Vec_t normal;
  bool valid;

  Precision cosa = 4 / std::sqrt(17.), sina = 1 / std::sqrt(17.);

  Vec_t trapvert[8] = {Vec_t(-10.0, -20.0, -40.0), Vec_t(+10.0, -20.0, -40.0), Vec_t(-10.0, +20.0, -40.0),
                       Vec_t(+10.0, +20.0, -40.0), Vec_t(-30.0, -40.0, +40.0), Vec_t(+30.0, -40.0, +40.0),
                       Vec_t(-30.0, +40.0, +40.0), Vec_t(+30.0, +40.0, +40.0)};

  Trap_t trap1("Test Boxlike #1", 40, 0, 0, 30, 20, 20, 0, 30, 20, 20, 0); // box:20,30,40

  //    Trap_t trap2("Test Trdlike #2",40,0,0,20,10,10,0,40,30,30,0);

  Trap_t trap2("Test Trdlike #2", trapvert);

  Trap_t trap3("trap3", 50, 0, 0, 50, 50, 50, kPi / 4, 50, 50, 50, kPi / 4);
  Trap_t trap4("trap4", 50, 0, 0, 50, 50, 50, -kPi / 4, 50, 50, 50, -kPi / 4);

  // confirm Trd calculations
  Trap_t trap5("Trd2", 20, 20, 30, 30, 40);
  // std::cout<<" Trd-like capacity: "<< trap5.Capacity() <<"\n";
  // std::cout<<" Trd-like surface area: "<< trap5.SurfaceArea() <<"\n";

  Trap_t largeTrap("Large shallow side exit", 1.e20, 0., 0., 1.e20, 1.e6, 1.e6, 0., 1.e20, 1.e6, 1.e6, 0.);
  Vec_t shallowXExit(1.e-12, 1., 0.);
  Dist = largeTrap.DistanceToOut(pzero, shallowXExit);
  VECGEOM_ASSERT(Dist > 0.9e18 && Dist < 1.1e18);

  Precision alpha = 1.55;
  vecgeom::UnplacedTrapezoid alphaTrap(1.e6, 1.e20, 1.e20, alpha, 0., 0.);
  Precision expectedInvProjectionScale = 1. / (1.e20 * (1. + 2. * vecgeom::Abs(vecgeom::Tan(alpha))));
  VECGEOM_ASSERT(ApproxEqual<Precision>(alphaTrap.GetStruct().fInvProjectionScale, expectedInvProjectionScale));

  Vec_t Corners[8];
  Corners[0] = Vec_t(-3., -3., -3.);
  Corners[1] = Vec_t(3., -3., -3.);
  Corners[2] = Vec_t(-3., 3., -3.);
  Corners[3] = Vec_t(3., 3., -3.);
  Corners[4] = Vec_t(-3., -3., 3.);
  Corners[5] = Vec_t(3., -3., 3.);
  Corners[6] = Vec_t(-3., 3., 3.);
  Corners[7] = Vec_t(3., 3., 3.);

  Trap_t tempTrap("temp trap", Corners);

  // Check cubic volume

  vol      = trap1.Capacity();
  volCheck = 8 * 20 * 30 * 40;
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  vol      = trap4.Capacity();
  volCheck = 8 * 50 * 50 * 50;
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  vol      = trap3.Capacity();
  volCheck = 8 * 50 * 50 * 50;
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  vol      = trap2.Capacity();
  volCheck = 2 * 40. * ((20. + 40.) * (10. + 30.) + (30. - 10.) * (40. - 20.) / 3.);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check surface area

  vol      = trap1.SurfaceArea();
  volCheck = 2 * (40 * 60 + 80 * 60 + 80 * 40);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  vol      = trap2.SurfaceArea();
  volCheck = 4 * (10 * 20 + 30 * 40) + 2 * ((20 + 40) * std::sqrt(4 * 40 * 40 + (30 - 10) * (30 - 10)) +
                                            (30 + 10) * std::sqrt(4 * 40 * 40 + (40 - 20) * (40 - 20)));
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // std::cout<<"Trd Surface Area : " << trap5.SurfaceArea()<<std::endl;
  VECGEOM_ASSERT(trap5.SurfaceArea() == 20800);

  // vecgeom::cxx::SimpleTrapezoid const* ptrap1 = dynamic_cast<vecgeom::cxx::SimpleTrapezoid*>(&trap1);
  // if(ptrap1 != NULL) {
  // ptrap1->Print();
  // // ptrap1->PrintType(std::cout);
  // }

  // Check Inside
  VECGEOM_ASSERT(trap1.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(trap1.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(trap1.Inside(ponxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trap1.Inside(ponyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trap1.Inside(ponzside) == vecgeom::EInside::kSurface);

  VECGEOM_ASSERT(trap2.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(trap2.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(trap2.Inside(ponxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trap2.Inside(ponyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trap2.Inside(ponzside) == vecgeom::EInside::kSurface);

  // test SamplePointOnSurface()
  vecgeom::Vector3D<vecgeom::Precision> ponsurf;
  for (int i = 0; i < 100000; ++i) {
    ponsurf = trap1.GetUnplacedVolume()->SamplePointOnSurface();
    VECGEOM_ASSERT(trap1.Inside(ponsurf) == vecgeom::EInside::kSurface);
    VECGEOM_ASSERT(trap1.Normal(ponsurf, normal) && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  }
  for (int i = 0; i < 100000; ++i) {
    ponsurf = trap2.GetUnplacedVolume()->SamplePointOnSurface();
    VECGEOM_ASSERT(trap2.Inside(ponsurf) == vecgeom::EInside::kSurface);
    VECGEOM_ASSERT(trap2.Normal(ponsurf, normal) && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  }

  // Check Surface Normal

  valid = trap1.Normal(ponxside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(1., 0., 0.)));
  valid = trap1.Normal(ponmxside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-1., 0., 0.)));
  valid = trap1.Normal(ponyside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0., 1., 0.)));
  valid = trap1.Normal(ponmyside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0., -1., 0.)));
  valid = trap1.Normal(ponzside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0., 0., 1.)));
  valid = trap1.Normal(ponmzside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0., 0., -1.)));
  valid = trap1.Normal(ponzsidey, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0., 0., 1.)));
  valid = trap1.Normal(ponmzsidey, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0., 0., -1.)));

  // Normals on Edges

  Vec_t edgeXY(20.0, 30., 0.0);
  Vec_t edgemXmY(-20.0, -30., 0.0);
  Vec_t edgeXmY(20.0, -30., 0.0);
  Vec_t edgemXY(-20.0, 30., 0.0);
  Vec_t edgeXZ(20.0, 0.0, 40.0);
  Vec_t edgemXmZ(-20.0, 0.0, -40.0);
  Vec_t edgeXmZ(20.0, 0.0, -40.0);
  Vec_t edgemXZ(-20.0, 0.0, 40.0);
  Vec_t edgeYZ(0.0, 30.0, 40.0);
  Vec_t edgemYmZ(0.0, -30.0, -40.0);
  Vec_t edgeYmZ(0.0, 30.0, -40.0);
  Vec_t edgemYZ(0.0, -30.0, 40.0);

  // Precision invSqrt2 = 1.0 / std::sqrt(2.0);
  // Precision invSqrt3 = 1.0 / std::sqrt(3.0);

  valid = trap1.Normal(edgeXY, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt2, invSqrt2, 0.0)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));

  valid = trap1.Normal(edgemXmY, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt2, -invSqrt2, 0.0)) && valid);
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgeXmY, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt2, -invSqrt2, 0.0)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemXY, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt2, invSqrt2, 0.0)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));

  valid = trap1.Normal(edgeXZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt2, 0.0, invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemXmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, -invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgeXmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt2, 0.0, -invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemXZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));

  valid = trap1.Normal(edgeYZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0.0, invSqrt2, invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemYmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0.0, -invSqrt2, -invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgeYmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0.0, invSqrt2, -invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemYZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0.0, -invSqrt2, invSqrt2)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));

  // Normals on corners

  Vec_t cornerXYZ(20.0, 30., 40.0);
  Vec_t cornermXYZ(-20.0, 30., 40.0);
  Vec_t cornerXmYZ(20.0, -30., 40.0);
  Vec_t cornermXmYZ(-20.0, -30., 40.0);
  Vec_t cornerXYmZ(20.0, 30., -40.0);
  Vec_t cornermXYmZ(-20.0, 30., -40.0);
  Vec_t cornerXmYmZ(20.0, -30., -40.0);
  Vec_t cornermXmYmZ(-20.0, -30., -40.0);

  valid = trap1.Normal(cornerXYZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt3, invSqrt3, invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXYZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt3, invSqrt3, invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornerXmYZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt3, -invSqrt3, invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXmYZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt3, -invSqrt3, invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornerXYmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt3, invSqrt3, -invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXYmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt3, invSqrt3, -invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornerXmYmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(invSqrt3, -invSqrt3, -invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXmYmZ, normal);
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-invSqrt3, -invSqrt3, -invSqrt3)));
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));

  valid = trap2.Normal(ponxside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(cosa, 0, -sina)));
  valid = trap2.Normal(ponmxside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));
  valid = trap2.Normal(ponyside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, cosa, -sina)));
  valid = trap2.Normal(ponmyside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, -cosa, -sina)));
  valid = trap2.Normal(ponzside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trap2.Normal(ponmzside, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, -1)));
  valid = trap2.Normal(ponzsidey, normal);
  VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trap2.Normal(ponmzsidey, normal);
  // std::cout << " Normal at " << ponmzsidey << " is " << normal
  //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::endl;
  // VECGEOM_ASSERT(valid && ApproxEqual(normal, Vec_t(0, 0.615412, -0.788205))); // (0,cosa,-sina) ?
  VECGEOM_ASSERT(valid && ApproxEqual<Precision>(normal.Mag2(), 1.0));

  // SafetyToOut(P)

  Dist = trap1.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = trap1.SafetyToOut(vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 19));
  Dist = trap1.SafetyToOut(vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = trap1.SafetyToOut(vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));

  Dist = trap2.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20 * cosa));
  Dist = trap2.SafetyToOut(vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 19 * cosa));
  Dist = trap2.SafetyToOut(vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20 * cosa));
  Dist = trap2.SafetyToOut(vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20 * cosa + sina));

  // DistanceToOut(P,V)

  Dist  = trap1.DistanceToOut(pzero, vx);
  valid = trap1.Normal(pzero + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20) && ApproxEqual(normal, vx));

  Dist  = trap1.DistanceToOut(pzero, vmx);
  valid = trap1.Normal(pzero + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20) && ApproxEqual(normal, vmx));

  Dist  = trap1.DistanceToOut(pzero, vy);
  valid = trap1.Normal(pzero + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30) && ApproxEqual(normal, vy));

  Dist  = trap1.DistanceToOut(pzero, vmy);
  valid = trap1.Normal(pzero + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30) && ApproxEqual(normal, vmy));

  Dist  = trap1.DistanceToOut(pzero, vz);
  valid = trap1.Normal(pzero + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vz));

  Dist  = trap1.DistanceToOut(pzero, vmz);
  valid = trap1.Normal(pzero + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vmz));

  Dist  = trap1.DistanceToOut(pzero, vxy);
  valid = trap1.Normal(pzero + Dist * vxy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, std::sqrt(800.)) && ApproxEqual(normal, vx));

  Dist  = trap1.DistanceToOut(ponxside, vx);
  valid = trap1.Normal(ponxside + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vx));

  Dist  = trap1.DistanceToOut(ponmxside, vmx);
  valid = trap1.Normal(ponmxside + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vmx));

  Dist  = trap1.DistanceToOut(ponyside, vy);
  valid = trap1.Normal(ponyside + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vy));

  Dist  = trap1.DistanceToOut(ponmyside, vmy);
  valid = trap1.Normal(ponmyside + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vmy));

  Dist  = trap1.DistanceToOut(ponzside, vz);
  valid = trap1.Normal(ponzside + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vz));

  Dist  = trap1.DistanceToOut(ponmzside, vmz);
  valid = trap1.Normal(ponmzside + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vmz));

  Dist = trap1.DistanceToOut(ponxside, vmx);
  // std::cout<<"Line "<<__LINE__<<": trap1.D2O(): pt="<< ponxside <<", dir="<< vmx <<", dist="<< Dist <<"\n";
  valid = trap1.Normal(ponxside + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vmx));

  Dist  = trap1.DistanceToOut(ponmxside, vx);
  valid = trap1.Normal(ponmxside + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vx));

  Dist  = trap1.DistanceToOut(ponyside, vmy);
  valid = trap1.Normal(ponyside + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60) && ApproxEqual(normal, vmy));

  Dist  = trap1.DistanceToOut(ponmyside, vy);
  valid = trap1.Normal(ponmyside + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60) && ApproxEqual(normal, vy));

  Dist  = trap1.DistanceToOut(ponzside, vmz);
  valid = trap1.Normal(ponzside + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80) && ApproxEqual(normal, vmz));

  Dist  = trap1.DistanceToOut(ponmzside, vz);
  valid = trap1.Normal(ponmzside + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80) && ApproxEqual(normal, vz));

  Dist  = trap2.DistanceToOut(pzero, vx);
  valid = trap2.Normal(pzero + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20) && ApproxEqual(normal, Vec_t(cosa, 0, -sina)));

  Dist  = trap2.DistanceToOut(pzero, vmx);
  valid = trap2.Normal(pzero + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20) && ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));

  Dist  = trap2.DistanceToOut(pzero, vy);
  valid = trap2.Normal(pzero + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30) && ApproxEqual(normal, Vec_t(0, cosa, -sina)));

  Dist  = trap2.DistanceToOut(pzero, vmy);
  valid = trap2.Normal(pzero + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30) && ApproxEqual(normal, Vec_t(0, -cosa, -sina)));

  Dist  = trap2.DistanceToOut(pzero, vz);
  valid = trap2.Normal(pzero + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vz));

  Dist  = trap2.DistanceToOut(pzero, vmz);
  valid = trap2.Normal(pzero + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vmz));

  Dist  = trap2.DistanceToOut(pzero, vxy);
  valid = trap2.Normal(pzero + Dist * vxy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, std::sqrt(800.)));

  Dist  = trap2.DistanceToOut(ponxside, vx);
  valid = trap2.Normal(ponxside + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(cosa, 0, -sina)));

  Dist  = trap2.DistanceToOut(ponmxside, vmx);
  valid = trap2.Normal(ponmxside + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));

  Dist  = trap2.DistanceToOut(ponyside, vy);
  valid = trap2.Normal(ponyside + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(0, cosa, -sina)));

  Dist  = trap2.DistanceToOut(ponmyside, vmy);
  valid = trap2.Normal(ponmyside + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(0, -cosa, -sina)));

  Dist  = trap2.DistanceToOut(ponzside, vz);
  valid = trap2.Normal(ponzside + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vz));

  Dist  = trap2.DistanceToOut(ponmzside, vmz);
  valid = trap2.Normal(ponmzside + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vmz));

  // SafetyToIn(P)

  Dist = trap1.SafetyToIn(pbig);
  // std::cout<<"trap1.SafetyToIn() = point="<< pbig <<", safety="<< Dist <<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));

  Dist = trap1.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));

  Dist = trap1.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));

  Dist = trap1.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));

  Dist = trap1.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));

  Dist = trap1.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));

  Dist = trap1.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));

  Dist = trap2.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80 * cosa));
  Dist = trap2.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80 * cosa));
  Dist = trap2.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70 * cosa));
  Dist = trap2.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70 * cosa));
  Dist = trap2.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trap2.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));

  //=== add test cases to reproduce a crash in Geant4: negative SafetyToOut() is not acceptable
  // std::cout <<"trap1.S2O(): Line "<< __LINE__ <<", p="<< testp <<", saf2out=" << Dist <<"\n";

  Vec_t testp;
  Precision testValue = 0.11;
  testp               = ponxside + testValue * vx;
  Dist                = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
  Dist = trap1.SafetyToOut(testp);
  if (Dist > 0) std::cout << "trap1.S2O(): Line " << __LINE__ << ", p=" << testp << ", saf2out=" << Dist << "\n";
  VECGEOM_ASSERT(Dist <= 0);

  testp = ponxside - testValue * vx;
  Dist  = trap1.SafetyToIn(testp);
  if (Dist > 0) std::cout << "trap1.S2I(): Line " << __LINE__ << ", p=" << testp << ", saf2in=" << Dist << "\n";
  VECGEOM_ASSERT(Dist <= 0);
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponmxside + testValue * vx;
  Dist  = trap1.SafetyToIn(testp);
  if (Dist > 0) std::cout << "trap1.S2I(): Line " << __LINE__ << ", p=" << testp << ", saf2in=" << Dist << "\n";
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(Dist >= 0);

  testp = ponmxside - testValue * vx;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(Dist <= 0);

  testp = ponyside + testValue * vy;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(Dist <= 0);

  testp = ponyside - testValue * vy;
  Dist  = trap1.SafetyToIn(testp);
  Dist  = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponmyside + testValue * vy;
  Dist  = trap1.SafetyToIn(testp);
  Dist  = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponmyside - testValue * vy;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(Dist <= 0);

  testp = ponzside + testValue * vz;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(Dist <= 0);

  testp = ponzside - testValue * vz;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(Dist <= 0);
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponmzside + testValue * vz;
  Dist  = trap1.SafetyToIn(testp);
  // std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
  VECGEOM_ASSERT(Dist <= 0.);
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));

  testp = ponmzside - testValue * vz;
  Dist  = trap1.SafetyToIn(testp);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, testValue));
  Dist = trap1.SafetyToOut(testp);
  VECGEOM_ASSERT(Dist <= 0);

  // DistanceToIn(P,V)

  Dist = trap1.DistanceToIn(pbigx, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trap1.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trap1.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trap1.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trap1.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trap1.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trap1.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = trap1.DistanceToIn(pbigmx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  Dist = trap2.DistanceToIn(pbigx, vmx);
  // std::cout<<"Line "<< __LINE__ <<", D2I(): point="<< pbigx <<", dir="<< vmx <<" -> dist2in="<< Dist <<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trap2.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trap2.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trap2.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trap2.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trap2.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trap2.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = trap2.DistanceToIn(pbigmx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  dist = trap3.DistanceToIn(Vec_t(50, -50, 0), vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, 50));

  dist = trap3.DistanceToIn(Vec_t(50, -50, 0), vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap4.DistanceToIn(Vec_t(50, 50, 0), vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap4.DistanceToIn(Vec_t(50, 50, 0), vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, 50));

  dist = trap1.DistanceToIn(Vec_t(0, 60, 0), vxmy);
  // std::cout<<" LIne "<< __LINE__ <<", trap1.D2I(): point=(0,60,0), dir="<< vxmy <<", d2in="<< dist <<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap1.DistanceToIn(Vec_t(0, 50, 0), vxmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, sqrt(800.)));

  dist = trap1.DistanceToIn(Vec_t(0, 40, 0), vxmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, 10.0 * std::sqrt(2.0)));

  dist = trap1.DistanceToIn(Vec_t(0, 40, 50), vxmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  // Parallel to side planes

  dist = trap1.DistanceToIn(Vec_t(40, 60, 0), vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap1.DistanceToIn(Vec_t(40, 60, 0), vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap1.DistanceToIn(Vec_t(40, 60, 50), vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap1.DistanceToIn(Vec_t(0, 0, 50), vymz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, 10.0 * std::sqrt(2.0)));

  dist = trap1.DistanceToIn(Vec_t(0, 0, 80), vymz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, kInfLength));

  dist = trap1.DistanceToIn(Vec_t(0, 0, 70), vymz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(dist, 30.0 * sqrt(2.0)));

  // Check Extent and cached BBox

  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  trap1.Extent(minExtent, maxExtent);
  trap1.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-20, -30, -40)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(20, 30, 40)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));
  trap2.Extent(minExtent, maxExtent);
  trap2.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-30, -40, -40)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(30, 40, 40)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));

#ifndef VECGEOM_NO_SPECIALIZATION
  Precision tdz = 5., ttheta = 0., tphi = 0., tdy1 = 4., tdx1 = 3., tdx2 = 3., tAlpha1 = 0., tdy2 = 4., tdx3 = 3.,
            tdx4 = 3., tAlpha2 = 0.;

  auto boxLikeTrap = vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedTrapezoid>(
      tdz, ttheta, tphi, tdy1, tdx1, tdx2, tAlpha1, tdy2, tdx3, tdx4, tAlpha2);
  tdx3 = 3.5;
  tdx4 = 3.5;

  auto trd1LikeTrap = vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedTrapezoid>(
      tdz, ttheta, tphi, tdy1, tdx1, tdx2, tAlpha1, tdy2, tdx3, tdx4, tAlpha2);
  tdy2 = 4.5;

  auto trd2LikeTrap = vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedTrapezoid>(
      tdz, ttheta, tphi, tdy1, tdx1, tdx2, tAlpha1, tdy2, tdx3, tdx4, tAlpha2);
  tdx3    = 3.;
  tdx4    = 3.;
  tdy2    = 4.;
  tAlpha1 = 30;
  tAlpha2 = 30;
  ttheta  = 30;
  tphi    = 45;

  auto parallelpipedLikeTrap = vecgeom::GeoManager::MakeInstance<vecgeom::cxx::UnplacedTrapezoid>(
      tdz, ttheta, tphi, tdy1, tdx1, tdx2, tAlpha1, tdy2, tdx3, tdx4, tAlpha2);

  using BoxLikeTrap  = vecgeom::SUnplacedImplAs<vecgeom::cxx::UnplacedTrapezoid, vecgeom::cxx::UnplacedBox>;
  using Trd1LikeTrap = vecgeom::SUnplacedImplAs<vecgeom::cxx::UnplacedTrapezoid,
                                                vecgeom::cxx::SUnplacedTrd<vecgeom::cxx::TrdTypes::Trd1>>;
  using Trd2LikeTrap = vecgeom::SUnplacedImplAs<vecgeom::cxx::UnplacedTrapezoid,
                                                vecgeom::cxx::SUnplacedTrd<vecgeom::cxx::TrdTypes::Trd2>>;
  using ParallelepipedLikeTrap =
      vecgeom::SUnplacedImplAs<vecgeom::cxx::UnplacedTrapezoid, vecgeom::cxx::UnplacedParallelepiped>;
  // Checking type of boxLikeTrap, it should return true with pointer of BoxLikeTrap
  VECGEOM_ASSERT(dynamic_cast<BoxLikeTrap *>(boxLikeTrap));
  VECGEOM_ASSERT(dynamic_cast<Trd1LikeTrap *>(trd1LikeTrap));
  VECGEOM_ASSERT(dynamic_cast<Trd2LikeTrap *>(trd2LikeTrap));
  VECGEOM_ASSERT(dynamic_cast<ParallelepipedLikeTrap *>(parallelpipedLikeTrap));

#endif

  return true;
}

void TestVECGEOM375()
{
  // unit test coming from Issue VECGEOM-375
  // Note: angles in rad, lengths in mm!
  Precision alpha1inRad = 0.4997657953434789 * deg;
  Precision alpha2inRad = 0.4997660621477440 * deg;
  vecgeom::UnplacedTrapezoid t(/*pDz=*/1522, /*pTheta=*/1.03516586152568 * deg,
                               /* Precision pPhi=*/-90.4997606158588 * deg, /*pDy1=*/147.5,
                               /*pDx1=*/11.0548515319824, /*pDx2=*/13.62808513641355, alpha1inRad,
                               /*pDy2 = */ 92.5, /*pDx3=*/11.0548515319824, /*pDx4=*/12.6685743331909, alpha2inRad);

  vecgeom::LogicalVolume logvol("", &t);
  logvol.Place();
  // t.StreamInfo(std::cout);
  // t.Print(std::cout);

  using Vec_t = vecgeom::Vector3D<Precision>;
  Vec_t ponsurf, normal;
  // check normal for a million points
  for (int i = 0; i < 100000; ++i) {
    ponsurf = t.SamplePointOnSurface();
    if (t.Inside(ponsurf) != vecgeom::EInside::kSurface) {
      Precision dist = vecCore::math::Max(t.SafetyToIn(ponsurf), t.SafetyToOut(ponsurf));
      // if triggered, a quick fix is to relax the value of trapSurfaceTolerance in TrapezoidImplementation.h
      std::cout << "*** Not on surface: i=" << i << ", ponsurf=" << ponsurf << " for dist=" << dist << "\n";
    }
    VECGEOM_ASSERT(t.Inside(ponsurf) == vecgeom::EInside::kSurface);
    VECGEOM_ASSERT(t.Normal(ponsurf, normal) && ApproxEqual<Precision>(normal.Mag2(), 1.0));
  }
}

void TestVECGEOM353()
{
  // unit test coming from Issue VECGEOM-353
  vecgeom::UnplacedTrapezoid t(/*pDz=*/0.5, /*pTheta=*/0, /* Precision pPhi=*/0, /*pDy1=*/3,
                               /*pDx1=*/14.237500000000001, /* Precision pDx2 =*/12.592500000000001,
                               /*pTanAlpha1 =*/0.274166666666666, /*pDy2 = */ 3,
                               /*pDx3 = */ 14.237500000000001, /*pDx4 = */ 12.592500000000001,
                               /*pTanAlpha2 = */ 0.274166666666666);

  vecgeom::LogicalVolume l("", &t);
  auto p      = l.Place();
  using Vec_t = vecgeom::Vector3D<Precision>;
  Vec_t point(-16.483749999999997, -6.4512999999999989, 0.00000099999999999999995);
  VECGEOM_ASSERT(!p->Contains(point));
  auto dist = p->DistanceToIn(point, Vec_t(1., 0., 0.));
  // std::cout<<" TestVECGEOM353(): point="<< point <<", dir="<< Vec_t(1,0,0) <<" - distToIn="<< dist <<"\n";
  VECGEOM_ASSERT(dist == vecgeom::kInfLength);
}

void TestVECGEOM393()
{
  using namespace vecgeom;
  // unit test coming from Issue VECGEOM-393
  vecgeom::UnplacedTrapezoid trap(60, 20 * deg, 5 * deg, 40, 30, 40, 10 * deg, 16, 10, 14, 10 * deg);

  Vector3D<Precision> extMin, extMax;
  trap.Extent(extMin, extMax);
  VECGEOM_ASSERT(ApproxEqual(extMin, Vector3D<Precision>(-58.80819229, -41.90332577, -60.)));
  VECGEOM_ASSERT(ApproxEqual(extMax, Vector3D<Precision>(38.57634475, 38.09667423, 60.)));
}

int main(int argc, char *argv[])
{
  TestVECGEOM375();
  TestVECGEOM353();
  TestVECGEOM393();
  TestTrap<VECGEOM_NAMESPACE::SimpleTrapezoid>();
  std::cout << "VecGeom Trap passed.\n";

  return 0;
}
