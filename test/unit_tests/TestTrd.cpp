// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for Trd
/// @file test/unit_tests/TestTrd.cpp
/// @author Tatiana Nikitina

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Trd.h"
#include "ApproxEqual.h"
#include <cmath>
#include "VecGeom/management/GeoManager.h"

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

bool testvecgeom = true;

using namespace vecgeom;

template <class Trd_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestTrd()
{
  vecgeom::EnumInside inside;
  Vec_t pzero(0, 0, 0);
  Vec_t ponxside(20, 0, 0), ponyside(0, 30, 0), ponzside(0, 0, 40);
  Vec_t ponmxside(-20, 0, 0), ponmyside(0, -30, 0), ponmzside(0, 0, -40);
  Vec_t ponzsidey(0, 25, 40), ponmzsidey(0, 25, -40);

  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  double Dist, vol, volCheck;
  Vec_t normal;
  bool valid;

  Trd_t trd1("Test Box #1", 20, 20, 30, 30, 40);
  Trd_t trd2("Test Trd", 10, 30, 20, 40, 40);
  Trd_t trd3("BABAR Trd", 0.14999999999999999, 0.14999999999999999, 24.707000000000001, 24.707000000000001,
             22.699999999999999);

  // check Cubic volume

  vol      = trd1.Capacity();
  volCheck = 8 * 20 * 30 * 40;
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check Surface area

  // std::cout<<"Trd Surface Area : " << trd1.SurfaceArea()<<std::endl;
  VECGEOM_ASSERT(trd1.SurfaceArea() == 20800);

  // Check Inside

  VECGEOM_ASSERT(trd1.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(trd1.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(trd1.Inside(ponxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trd1.Inside(ponyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trd1.Inside(ponzside) == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(20, 30, 40));
  //  std::cout << "trd1.Inside((20,30,40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(-20, 30, 40));
  // std::cout << "trd1.Inside((-20,30,40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(20, -30, 40));
  //  std::cout << "trd1.Inside((20,-30,40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(20, 30, -40));
  // std::cout << "trd1.Inside((20,30,-40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(20, 30, 0));
  // std::cout << "trd1.Inside((20,30,0)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(0, 30, 40));
  // std::cout << "trd1.Inside((0,30,40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(20, 0, 40));
  // std::cout << "trd1.Inside((20,0,40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = trd1.Inside(Vec_t(-20, -30, -40));
  // std::cout << "trd1.Inside((-20,-30,-40)) = " << OutputInside(inside) << std::ensl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  VECGEOM_ASSERT(trd2.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(trd2.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(trd2.Inside(ponxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trd2.Inside(ponyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(trd2.Inside(ponzside) == vecgeom::EInside::kSurface);

  // Check Surface Normal

  valid = trd1.Normal(ponxside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(1, 0, 0)));
  valid = trd1.Normal(ponmxside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-1, 0, 0)));
  valid = trd1.Normal(ponyside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 1, 0)));
  valid = trd1.Normal(ponmyside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, -1, 0)));
  valid = trd1.Normal(ponzside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trd1.Normal(ponmzside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, -1)));
  valid = trd1.Normal(ponzsidey, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trd1.Normal(ponmzsidey, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, -1)));

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

  double invSqrt2 = 1.0 / std::sqrt(2.0);
  double invSqrt3 = 1.0 / std::sqrt(3.0);

  valid = trd1.Normal(edgeXY, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt2, invSqrt2, 0.0)) && valid);

  // std::cout << " Normal at " << edgeXY << " is " << normal
  //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::ensl;

  valid = trd1.Normal(edgemXmY, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt2, -invSqrt2, 0.0)));
  valid = trd1.Normal(edgeXmY, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt2, -invSqrt2, 0.0)));
  valid = trd1.Normal(edgemXY, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt2, invSqrt2, 0.0)));

  valid = trd1.Normal(edgeXZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt2, 0.0, invSqrt2)));
  valid = trd1.Normal(edgemXmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, -invSqrt2)));
  valid = trd1.Normal(edgeXmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt2, 0.0, -invSqrt2)));
  valid = trd1.Normal(edgemXZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, invSqrt2)));

  valid = trd1.Normal(edgeYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.0, invSqrt2, invSqrt2)));
  valid = trd1.Normal(edgemYmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.0, -invSqrt2, -invSqrt2)));
  valid = trd1.Normal(edgeYmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.0, invSqrt2, -invSqrt2)));
  valid = trd1.Normal(edgemYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.0, -invSqrt2, invSqrt2)));

  // Normals on corners

  Vec_t cornerXYZ(20.0, 30., 40.0);
  Vec_t cornermXYZ(-20.0, 30., 40.0);
  Vec_t cornerXmYZ(20.0, -30., 40.0);
  Vec_t cornermXmYZ(-20.0, -30., 40.0);
  Vec_t cornerXYmZ(20.0, 30., -40.0);
  Vec_t cornermXYmZ(-20.0, 30., -40.0);
  Vec_t cornerXmYmZ(20.0, -30., -40.0);
  Vec_t cornermXmYmZ(-20.0, -30., -40.0);

  valid = trd1.Normal(cornerXYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt3, invSqrt3, invSqrt3)));
  valid = trd1.Normal(cornermXYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt3, invSqrt3, invSqrt3)));
  valid = trd1.Normal(cornerXmYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt3, -invSqrt3, invSqrt3)));
  valid = trd1.Normal(cornermXmYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt3, -invSqrt3, invSqrt3)));
  valid = trd1.Normal(cornerXYmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt3, invSqrt3, -invSqrt3)));
  valid = trd1.Normal(cornermXYmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt3, invSqrt3, -invSqrt3)));
  valid = trd1.Normal(cornerXmYmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt3, -invSqrt3, -invSqrt3)));
  valid = trd1.Normal(cornermXmYmZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt3, -invSqrt3, -invSqrt3)));

  double cosa = 4 / std::sqrt(17.), sina = 1 / std::sqrt(17.);

  valid = trd2.Normal(ponxside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(cosa, 0, -sina)));
  valid = trd2.Normal(ponmxside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));
  valid = trd2.Normal(ponyside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, cosa, -sina)));
  valid = trd2.Normal(ponmyside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, -cosa, -sina)));
  valid = trd2.Normal(ponzside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trd2.Normal(ponmzside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, -1)));
  valid = trd2.Normal(ponzsidey, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trd2.Normal(ponmzsidey, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, -1))); // (0,cosa,-sina) ?

  // SafetyToOut(P)

  Dist = trd1.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = trd1.SafetyToOut(vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 19));
  Dist = trd1.SafetyToOut(vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = trd1.SafetyToOut(vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));

  Dist = trd2.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20 * cosa));
  Dist = trd2.SafetyToOut(vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 19 * cosa));
  Dist = trd2.SafetyToOut(vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20 * cosa));
  Dist = trd2.SafetyToOut(vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20 * cosa + sina));

  // DistanceToOut(P,V)

  Dist = trd1.DistanceToOut(pzero, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = trd1.DistanceToOut(pzero, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = trd1.DistanceToOut(pzero, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30));
  Dist = trd1.DistanceToOut(pzero, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30));
  Dist = trd1.DistanceToOut(pzero, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40));
  Dist = trd1.DistanceToOut(pzero, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40));
  Dist = trd1.DistanceToOut(pzero, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, std::sqrt(800.)));

  Dist = trd1.DistanceToOut(ponxside, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = trd1.DistanceToOut(ponmxside, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = trd1.DistanceToOut(ponyside, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = trd1.DistanceToOut(ponmyside, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = trd1.DistanceToOut(ponzside, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = trd1.DistanceToOut(ponmzside, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  Dist  = trd2.DistanceToOut(pzero, vx);
  valid = trd2.Normal(pzero + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20) && ApproxEqual(normal, Vec_t(cosa, 0, -sina)));

  Dist  = trd2.DistanceToOut(pzero, vmx);
  valid = trd2.Normal(pzero + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20) && ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));

  Dist  = trd2.DistanceToOut(pzero, vy);
  valid = trd2.Normal(pzero + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30) && ApproxEqual(normal, Vec_t(0, cosa, -sina)));

  Dist  = trd2.DistanceToOut(pzero, vmy);
  valid = trd2.Normal(pzero + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30) && ApproxEqual(normal, Vec_t(0, -cosa, -sina)));

  Dist  = trd2.DistanceToOut(pzero, vz);
  valid = trd2.Normal(pzero + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vz));

  Dist  = trd2.DistanceToOut(pzero, vmz);
  valid = trd2.Normal(pzero + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40) && ApproxEqual(normal, vmz));

  Dist  = trd2.DistanceToOut(pzero, vxy);
  valid = trd2.Normal(pzero + Dist * vxy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, std::sqrt(800.)));

  Dist  = trd2.DistanceToOut(ponxside, vx);
  valid = trd2.Normal(ponxside + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(cosa, 0, -sina)));

  Dist  = trd2.DistanceToOut(ponmxside, vmx);
  valid = trd2.Normal(ponmxside + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));

  Dist  = trd2.DistanceToOut(ponyside, vy);
  valid = trd2.Normal(ponyside + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(0, cosa, -sina)));

  Dist  = trd2.DistanceToOut(ponmyside, vmy);
  valid = trd2.Normal(ponmyside + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, Vec_t(0, -cosa, -sina)));

  Dist  = trd2.DistanceToOut(ponzside, vz);
  valid = trd2.Normal(ponzside + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vz));

  Dist  = trd2.DistanceToOut(ponmzside, vmz);
  valid = trd2.Normal(ponmzside + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vmz));

  // SafetyToIn(P)

  Dist = trd1.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trd1.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trd1.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trd1.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trd1.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trd1.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));

  Dist = trd2.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80 * cosa));
  Dist = trd2.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80 * cosa));
  Dist = trd2.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70 * cosa));
  Dist = trd2.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70 * cosa));
  Dist = trd2.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trd2.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));

  // DistanceToIn(P,V)

  Dist = trd1.DistanceToIn(pbigx, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trd1.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trd1.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trd1.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trd1.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trd1.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trd1.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = trd1.DistanceToIn(pbigmx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  Dist = trd2.DistanceToIn(pbigx, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trd2.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80));
  Dist = trd2.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trd2.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70));
  Dist = trd2.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trd2.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 60));
  Dist = trd2.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = trd2.DistanceToIn(pbigmx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  Dist = trd3.DistanceToIn(Vec_t(0.15000000000000185, -22.048743592955137, 2.4268539333219472),
                           Vec_t(-0.76165597579890043, 0.64364445891356026, -0.074515708658524193).Unit());

  //    std::cout<<"BABAR trd distance = "<<Dist<<std::ensl ;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.0));

  // return-value = 2.4415531753644804e-15

  // Check Extent and cached BBox
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  trd1.Extent(minExtent, maxExtent);
  trd1.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-20, -30, -40)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(20, 30, 40)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));
  trd2.Extent(minExtent, maxExtent);
  trd2.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-30, -40, -40)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(30, 40, 40)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));

  // Simple Unit Tests for Factory of Trd.
  // Trd_t trdBoxTrd1("Test Trd", 20, 20, 30, 40);
  auto trdBoxTrd1 = GeoManager::MakeInstance<UnplacedTrd>(20, 20, 30, 40);
  VECGEOM_ASSERT(trdBoxTrd1->dx1() == 20);
  VECGEOM_ASSERT(trdBoxTrd1->dx2() == 20);
  VECGEOM_ASSERT(trdBoxTrd1->dy1() == 30);
  VECGEOM_ASSERT(trdBoxTrd1->dy2() == 30);
  VECGEOM_ASSERT(trdBoxTrd1->dz() == 40);

  // Trd_t trdBoxTrd2("TestTrdBox", 20, 20, 30, 30, 40);
  auto trdBoxTrd2 = GeoManager::MakeInstance<UnplacedTrd>(20, 20, 30, 30, 40);
  VECGEOM_ASSERT(trdBoxTrd2->dx1() == 20);
  VECGEOM_ASSERT(trdBoxTrd2->dx2() == 20);
  VECGEOM_ASSERT(trdBoxTrd2->dy1() == 30);
  VECGEOM_ASSERT(trdBoxTrd2->dy2() == 30);
  VECGEOM_ASSERT(trdBoxTrd2->dz() == 40);

  // Trd_t trdType1("Test Trd", 20, 25, 30, 40);
  auto trdType1 = GeoManager::MakeInstance<UnplacedTrd>(20, 25, 30, 40);
  VECGEOM_ASSERT(trdType1->dx1() == 20);
  VECGEOM_ASSERT(trdType1->dx2() == 25);
  VECGEOM_ASSERT(trdType1->dy1() == 30);
  VECGEOM_ASSERT(trdType1->dy2() == 30);
  VECGEOM_ASSERT(trdType1->dz() == 40);

  // Trd_t trdType2("Test Trd", 20, 25, 30, 35, 40);
  auto trdType2 = GeoManager::MakeInstance<UnplacedTrd>(20, 25, 30, 35, 40);
  VECGEOM_ASSERT(trdType2->dx1() == 20);
  VECGEOM_ASSERT(trdType2->dx2() == 25);
  VECGEOM_ASSERT(trdType2->dy1() == 30);
  VECGEOM_ASSERT(trdType2->dy2() == 35);
  VECGEOM_ASSERT(trdType2->dz() == 40);

#ifndef VECGEOM_NO_SPECIALIZATION
  using trd1Box = SUnplacedImplAs<SUnplacedTrd<TrdTypes::Trd1>, UnplacedBox>;
  // Checking type of trdBoxTrd1, it should return true with pointer of trd1Box
  VECGEOM_ASSERT(dynamic_cast<trd1Box *>(trdBoxTrd1));
  // Checking type of other specialized Trd's with type of trdBoxTrd1, all should return false
  VECGEOM_ASSERT(!dynamic_cast<trd1Box *>(trdBoxTrd2));
  VECGEOM_ASSERT(!dynamic_cast<trd1Box *>(trdType1));
  VECGEOM_ASSERT(!dynamic_cast<trd1Box *>(trdType2));

  // Similar test for trdBoxTrd2
  using trd2Box = SUnplacedImplAs<SUnplacedTrd<TrdTypes::Trd2>, UnplacedBox>;
  // Checking type of trdBoxTrd2, it should return true with pointer of trd2Box
  VECGEOM_ASSERT(dynamic_cast<trd2Box *>(trdBoxTrd2));
  // Checking type of other specialized Trd's with type of trdBoxTrd2, all should return false
  VECGEOM_ASSERT(!dynamic_cast<trd2Box *>(trdBoxTrd1));
  VECGEOM_ASSERT(!dynamic_cast<trd2Box *>(trdType1));
  VECGEOM_ASSERT(!dynamic_cast<trd2Box *>(trdType2));
#endif

  return true;
}

int main(int argc, char *argv[])
{
  TestTrd<vecgeom::SimpleTrd>();
  std::cout << "VecGeom Trd passed\n";

  return 0;
}
