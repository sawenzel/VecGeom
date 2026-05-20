//
// File:    TestPolycone.cpp
// Purpose: Unit tests for the polycone
//

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "ApproxEqual.h"
#include <cmath>
#include "VecGeom/management/GeoManager.h"

using namespace vecgeom;

template <class Polycone_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestPolycone()
{
  Precision RMINVec[8]  = {30, 30, 0, 0, 0, 0, 40, 40};
  Precision RMAXVec[8]  = {70, 70, 70, 40, 40, 80, 80, 60};
  Precision Z_Values[8] = {-20, -10, -10, 0, 10, 20, 30, 40};
  Precision Phi_Values[2];
  Phi_Values[0]       = -10. * kPi / 180.;
  Phi_Values[1]       = 10. * kPi / 180.;
  Polycone_t *MyPCone = new Polycone_t("MyPCone", Phi_Values[0], Phi_Values[1], 8, Z_Values, RMINVec, RMAXVec);

  Precision RMIN[3]   = {0, 0, 0};
  Precision RMAX[3]   = {70, 70, 80};
  Precision Z_Val2[3] = {-10, 0, 10};
  Polycone_t Simple("SimpleTube+Cone", 0, 360. * kPi / 180., 3, Z_Val2, RMIN, RMAX);

  Precision z1[8]    = {-2935., -1899., -1899., -1899., 1899., 1899., 1899., 2935.};
  Precision rmin1[8] = {74., 34., 31., 31., 31., 31., 34., 74.};
  Precision rmax1[8] = {1233., 1233., 1233., 1233., 1233., 1233., 1233., 1233.};
  Polycone_t cms_TRAK("oldcms_TRAK", 0, 360. * kPi / 180., 8, z1, rmin1, rmax1);

  int numRZ          = 9;
  Precision rmin2[9] = {0., 10., 10., 5., 5., 10., 10., 2., 2.};
  Precision z2[9]    = {5., 7., 9., 11., 25., 27., 29., 31., 35.};
  // Test construction of RZ polycone
  GeoManager::MakeInstance<UnplacedPolycone>(0., kPi / 4, numRZ, rmin2, z2);

  int Nz = 4;
  // a few cones
  Precision rmin[] = {0.1, 0.0, 0.0, 0.4};
  Precision rmax[] = {1., 2., 2., 1.5};
  Precision z[]    = {-1, -0.5, 0.5, 2};

  //  UnplacedPolycone poly1(0.,                /* initial phi starting angle */
  //                         360. * kPi / 180., // kTwoPi,    /* total phi angle */
  //                         Nz,                /* number corners in r,z space */
  //                         z,                 /* z coordinates */
  //                         rmin,              /* r coordinate of these corners */
  //                         rmax);

  auto poly1 = GeoManager::MakeInstance<UnplacedPolycone>(0.,                /* initial phi starting angle */
                                                          360. * kPi / 180., // kTwoPi,    /* total phi angle */
                                                          Nz,                /* number corners in r,z space */
                                                          z,                 /* z coordinates */
                                                          rmin,              /* r coordinate of these corners */
                                                          rmax);

  // poly1->Print();

  // let's make external separate cones representing the sections
  SUnplacedCone<ConeTypes::UniversalCone> section0(rmin[0], rmax[0], rmin[1], rmax[1], (z[1] - z[0]) / 2., 0, kTwoPi);
  SUnplacedCone<ConeTypes::UniversalCone> section1(rmin[1], rmax[1], rmin[2], rmax[2], (z[2] - z[1]) / 2., 0, kTwoPi);
  SUnplacedCone<ConeTypes::UniversalCone> section2(rmin[2], rmax[2], rmin[3], rmax[3], (z[3] - z[2]) / 2., 0, kTwoPi);

  VECGEOM_ASSERT(poly1->GetNz() == 4);
  VECGEOM_ASSERT(poly1->GetNSections() == 3);
  VECGEOM_ASSERT(poly1->GetSectionIndex(-0.8) == 0);
  VECGEOM_ASSERT(poly1->GetSectionIndex(0.51) == 2);
  VECGEOM_ASSERT(poly1->GetSectionIndex(0.) == 1);
  VECGEOM_ASSERT(poly1->GetSectionIndex(-2.) == -1);
  VECGEOM_ASSERT(poly1->GetSectionIndex(3.) == -2);
  VECGEOM_ASSERT(poly1->GetStartPhi() == 0.);
  VECGEOM_ASSERT((std::fabs(poly1->GetDeltaPhi() - kTwoPi)) < 1e-10);

  VECGEOM_ASSERT(poly1->GetStruct().fZs[0] == z[0]);
  VECGEOM_ASSERT(poly1->GetStruct().fZs[poly1->GetNSections()] == z[Nz - 1]);
  VECGEOM_ASSERT(poly1->Capacity() > 0);
  VECGEOM_ASSERT(std::fabs(poly1->Capacity() - (section0.Capacity() + section1.Capacity() + section2.Capacity())) <
                 1e-6);

  // create a placed version
  VPlacedVolume const *placedpoly1 = (new LogicalVolume("poly1", poly1))->Place(new Transformation3D());

  // test contains/inside
  VECGEOM_ASSERT(placedpoly1->Contains(Vec_t(0., 0., 0.)) == true);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec_t(0., 0., -2.)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec_t(0., 0., -0.8)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec_t(0., 0., -1.8)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec_t(0., 0., 10)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec_t(0., 0., 1.8)) == false);

  // test DistanceToIn
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec_t(0., 0., -3.), Vec_t(0., 0., 1.)) == 2.5);
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec_t(0., 0., -2.), Vec_t(0., 0., -1.)) == kInfLength);
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec_t(0., 0., 3), Vec_t(0., 0., -1.)) == 2.5);
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec_t(0., 0., 3), Vec_t(0., 0., 1.)) == kInfLength);
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec_t(3., 0., 0), Vec_t(-1., 0., 0.)) == 1);
  VECGEOM_ASSERT(std::fabs(placedpoly1->DistanceToIn(Vec_t(0., 0., 1.9999999), Vec_t(1., 0., 0.)) - 0.4) <
                 1000. * kTolerance);
  VECGEOM_ASSERT(Simple.DistanceToIn(Vec_t(25., 0., -10.), Vec_t(1., 0., 0.)) == kInfLength);
  VECGEOM_ASSERT(Simple.DistanceToIn(Vec_t(25., 0., 10.), Vec_t(1., 0., 0.)) == kInfLength);

  Precision zShared[3]    = {-1., 0., 1.};
  Precision rminShared[3] = {0., 0., 0.};
  Precision rmaxShared[3] = {1., 2., 1.};
  Polycone_t sharedPlaneRadialHit("SharedPlaneRadialHit", 0., kTwoPi, 3, zShared, rminShared, rmaxShared);
  Vec_t radialEntryDirection(-1., 0., 0.1);
  radialEntryDirection.Normalize();
  // The ray hits the outer conical surface exactly at the shared z plane. This is a radial hit, not a z-annulus
  // hand-off, so DistanceToIn must not reject it as an internal section transition.
  Precision radialEntry = sharedPlaneRadialHit.DistanceToIn(Vec_t(3., 0., -0.1), radialEntryDirection);
  VECGEOM_ASSERT(std::fabs(radialEntry - std::sqrt(1.01)) < 1000. * kTolerance);

  Precision rmaxSharedFlat[3] = {1., 2., 2.};
  Polycone_t sharedPlaneEdgeMiss("SharedPlaneEdgeMiss", 0., kTwoPi, 3, zShared, rminShared, rmaxSharedFlat);
  Vec_t edgeMissDirection(0.5, 0., 1.);
  edgeMissDirection.Normalize();
  // The ray touches the next section endpoint at its outer radius, but keeps
  // moving outside the following cylindrical section. This is a pure grazing
  // edge miss, not a valid DistanceToIn entry.
  VECGEOM_ASSERT(sharedPlaneEdgeMiss.DistanceToIn(Vec_t(1.5, 0., -1.), edgeMissDirection) == kInfLength);

  SUnplacedCone<ConeTypes::UniversalCone> simpleLowerSection(0., 70., 0., 70., 5., 0., kTwoPi);
  SUnplacedCone<ConeTypes::UniversalCone> simpleUpperSection(0., 70., 0., 80., 5., 0., kTwoPi);
  VECGEOM_ASSERT(simpleLowerSection.DistanceToIn(Vec_t(25., 0., -5.), Vec_t(1., 0., 0.)) == kInfLength);
  VECGEOM_ASSERT(std::fabs(simpleLowerSection.DistanceToOut(Vec_t(25., 0., -5.), Vec_t(1., 0., 0.)) - 45.) <
                 1000. * kTolerance);
  VECGEOM_ASSERT(simpleUpperSection.DistanceToIn(Vec_t(25., 0., 5.), Vec_t(1., 0., 0.)) == kInfLength);
  VECGEOM_ASSERT(std::fabs(simpleUpperSection.DistanceToOut(Vec_t(25., 0., 5.), Vec_t(1., 0., 0.)) - 55.) <
                 1000. * kTolerance);

  Precision zSharp[4]    = {-120., -20., -20., 120.};
  Precision rminSharp[4] = {0., 0., 0., 0.};
  Precision rmaxSharp[4] = {15., 15., 70., 70.};
  Polycone_t sharpJump("SharpJump", 0., kTwoPi, 4, zSharp, rminSharp, rmaxSharp);
  VECGEOM_ASSERT(std::fabs(sharpJump.DistanceToIn(Vec_t(80., 0., -20.), Vec_t(-1., 0., 0.)) - 65.) <
                 1000. * kTolerance);
  Vec_t grazingUpEntry(-1., 0., 1.e-10);
  grazingUpEntry.Normalize();
  VECGEOM_ASSERT(std::fabs(sharpJump.DistanceToIn(Vec_t(80., 0., -20.), grazingUpEntry) - 10.) < 1000. * kTolerance);

  // test SafetyToIn
  VECGEOM_ASSERT(placedpoly1->SafetyToIn(Vec_t(0., 0., -3.)) == 2.);
  VECGEOM_ASSERT(placedpoly1->SafetyToIn(Vec_t(0.5, 0., -1.)) == 0.);
  VECGEOM_ASSERT(placedpoly1->SafetyToIn(Vec_t(0., 0., 3)) == 1);
  VECGEOM_ASSERT(placedpoly1->SafetyToIn(Vec_t(2., 0., 0.1)) == 0);

  // test SafetyToOut
  VECGEOM_ASSERT(placedpoly1->SafetyToOut(Vec_t(0., 0., 0.)) == 0.5);
  VECGEOM_ASSERT(placedpoly1->SafetyToOut(Vec_t(0., 0., 0.5)) == 0.);
  VECGEOM_ASSERT(std::fabs(placedpoly1->SafetyToOut(Vec_t(1.9, 0., 0.0)) - 0.1) < 1000. * kTolerance);
  VECGEOM_ASSERT(placedpoly1->SafetyToOut(Vec_t(0.2, 0., -1)) == 0.);
  VECGEOM_ASSERT(placedpoly1->SafetyToOut(Vec_t(1.4, 0., 2)) == 0.);

  // test DistanceToOut
  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(0., 0., 0.), Vec_t(0., 0., 1.)) == 0.5);
  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(0., 0., 0.), Vec_t(0., 0., -1.)) == 0.5);
  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)) == 0.);
  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(2., 0., 0.), Vec_t(-1., 0., 0.)) == 4.);

  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(1., 0., 2), Vec_t(0., 0., 1.)) == 0.);
  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(0.5, 0., -1), Vec_t(0., 0., -1.)) == 0.);
  VECGEOM_ASSERT(placedpoly1->DistanceToOut(Vec_t(0.5, 0., -1), Vec_t(0., 0., 1.)) == 3.);

  // Check Cubic volume
  Precision vol, volCheck;
  vol      = Simple.Capacity();
  volCheck = kPi * (70 * 70 * 10 + 10 * (70 * 70 + 80 * 80 + 70 * 80) / 3.);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check Surface area
  vol      = Simple.SurfaceArea();
  volCheck = kPi * (70 * 70 + 80 * 80 + (70 + 80) * std::sqrt(10 * 10 + 10 * 10) + 10 * 2 * 70);
  VECGEOM_ASSERT(ApproxEqual<Precision>(vol, volCheck));

  // Check Inside
  Vec_t pzero(0, 0, 0);
  Vec_t ponxside(70, 0, -5), ponyside(0, 70, -5), ponzside(70, 0, 10);
  Vec_t ponmxside(-70, 0, -5), ponmyside(0, -70, -5), ponmzside(0, 0, -10);
  Vec_t ponzsidey(0, 25, 0), ponmzsidey(4, 25, 0);

  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Precision Dist;
  Vec_t normal;
  bool valid;
  VECGEOM_ASSERT(Simple.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(Simple.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(Simple.Inside(pbigx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(Simple.Inside(pbigy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(Simple.Inside(ponxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(ponyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(ponzside) == vecgeom::EInside::kSurface);

  VECGEOM_ASSERT(Simple.Inside(ponmxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(ponmyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(ponmzside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(ponzsidey) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(Simple.Inside(ponmzsidey) == vecgeom::EInside::kInside);

  VECGEOM_ASSERT(std::fabs(Simple.DistanceToOut(Vec_t(25., 0., -10.), Vec_t(1., 0., 0.)) - 45.) < 1000. * kTolerance);
  VECGEOM_ASSERT(std::fabs(Simple.DistanceToOut(Vec_t(25., 0., 10.), Vec_t(1., 0., 0.)) - 55.) < 1000. * kTolerance);

  // check that Normal() returns valid=false and a non-zero normal for points away from the surface

  Vec_t point(70, 70, -5);
  if ((valid = Simple.Normal(point, normal)) || !ApproxEqual<Precision>(normal.Mag2(), 1))
    std::cout << "Simple.Normal() normal not checked: Line " << __LINE__ << ", p=" << point << ", normal=" << normal
              << ", valid=" << valid << "\n";
  point.z() = -10;
  if ((valid = Simple.Normal(point, normal)) || !ApproxEqual<Precision>(normal.Mag2(), 1))
    std::cout << "Simple.Normal() normal not checked: Line " << __LINE__ << ", p=" << point << ", normal=" << normal
              << ", valid=" << valid << "\n";
  if ((valid = Simple.Normal(pbigz, normal)) || !ApproxEqual<Precision>(normal.Mag2(), 1))
    std::cout << "Simple.Normal() normal not checked: Line " << __LINE__ << ", p=" << pbigz << ", normal=" << normal
              << ", valid=" << valid << "\n";
  if ((valid = Simple.Normal(pbigmz, normal)) || !ApproxEqual<Precision>(normal.Mag2(), 1))
    std::cout << "Simple.Normal() normal not checked: Line " << __LINE__ << ", p=" << pbigmz << ", normal=" << normal
              << ", valid=" << valid << "\n";

  // Check Surface Normal

  valid = Simple.Normal(ponxside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(1, 0, 0)) && valid);
  valid = Simple.Normal(ponmxside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-1, 0, 0)));
  valid = Simple.Normal(ponyside, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 1, 0)));
  valid = Simple.Normal(Vec_t(0, 0, 10), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = Simple.Normal(Vec_t(0, 0, -10), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, -1)));

  // Normals on Edges

  Vec_t edgeXZ(80.0, 0.0, 10.0);
  Vec_t edgeYZ(0., 80.0, 10.0);
  Vec_t edgeXmZ(70.0, 0.0, -10.0);
  Vec_t edgeYmZ(0.0, 70.0, -10.0);
  Vec_t edgemXZ(-80.0, 0.0, 10.0);
  Vec_t edgemYZ(0., -80.0, 10.0);
  Vec_t edgemXmZ(-70.0, 0.0, -10.0);
  Vec_t edgemYmZ(0.0, -70.0, -10.0);
  // Precision invSqrt2 = 1.0 / std::sqrt(2.0);
  // Precision invSqrt3 = 1.0 / std::sqrt( 3.0);

  valid = Simple.Normal(edgeXmZ, normal);
  // VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(invSqrt2, 0.0, -invSqrt2)));
  valid = Simple.Normal(edgemXmZ, normal);
  // VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, -invSqrt2)));
  valid = Simple.Normal(edgeYmZ, normal);
  // VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.0, invSqrt2, -invSqrt2)));
  valid = Simple.Normal(edgemYmZ, normal);
  // VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.0, -invSqrt2, -invSqrt2)));

  const Precision xyn = 0.92388, zn = 0.382683;
  valid = Simple.Normal(edgeXZ, normal);
  std::cout << "Simple.Normal(): p=" << edgeXZ << ", normal=" << normal << ", valid=" << valid << std::endl;
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(xyn, 0, zn)));
  valid = Simple.Normal(edgemXZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-xyn, 0, zn)));
  valid = Simple.Normal(edgeYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, xyn, zn)));
  valid = Simple.Normal(edgemYZ, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, -xyn, zn)));

  // SafetyToOut(P)
  Dist = Simple.SafetyToOut(Vec_t(5, 5, -5));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 5));
  Dist = Simple.SafetyToOut(Vec_t(5, 5, 7));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 3));
  Dist = Simple.SafetyToOut(Vec_t(69, 0, -5));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 1));
  Dist = Simple.SafetyToOut(Vec_t(-3, -3, 8));
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 2));

  // DistanceToOut(P,V)

  Dist  = Simple.DistanceToOut(pzero, vx);
  valid = Simple.Normal(pzero + Dist * vx, normal);
  std::cout << "D2O normal not checked: Line " << __LINE__ << ", p=" << pzero << ", dir=" << vx << ", norm=" << normal
            << "\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70)); // && ApproxEqual(normal,vx));
  Dist  = Simple.DistanceToOut(pzero, vmx);
  valid = Simple.Normal(pzero + Dist * vmx, normal);
  std::cout << "D2O normal not checked: Line " << __LINE__ << ", p=" << pzero << ", dir=" << vmx << ", norm=" << normal
            << "\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70)); // && ApproxEqual(normal,vmx));
  Dist  = Simple.DistanceToOut(pzero, vy);
  valid = Simple.Normal(pzero + Dist * vy, normal);
  std::cout << "D2O normal not checked: Line " << __LINE__ << ", p=" << pzero << ", dir=" << vy << ", norm=" << normal
            << "\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70)); // &&ApproxEqual(normal,vy));
  Dist  = Simple.DistanceToOut(pzero, vmy);
  valid = Simple.Normal(pzero + Dist * vmy, normal);
  std::cout << "D2O normal not checked: Line " << __LINE__ << ", p=" << pzero << ", dir=" << vmy << ", norm=" << normal
            << "\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 70)); // &&ApproxEqual(normal,vmy));
  Dist  = Simple.DistanceToOut(pzero, vz);
  valid = Simple.Normal(pzero + Dist * vz, normal);
  // std::cout<<Dist<< " " <<norm<<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 10) && ApproxEqual(normal, vz));
  Dist  = Simple.DistanceToOut(Vec_t(70, 0, -10), vx);
  valid = Simple.Normal(Vec_t(70, 0, -10) + Dist * vx, normal);
  std::cout << "D2O normal not checked: Line " << __LINE__ << ", p=" << pzero << ", dir=" << vx << ", dist=" << Dist
            << ", norm=" << normal << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,0)&&ApproxEqual(normal,(vx-vz)/(vx-vz).Mag()));
  Dist = Simple.DistanceToOut(Vec_t(-70, 0, -1), vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0)); // && ApproxEqual(normal, vmx));
  Dist = Simple.DistanceToOut(Vec_t(0, 70, -10), vy);
  std::cout << "D2O normal not checked: Line " << __LINE__ << ", p=" << Vec_t(0, 70, -10) << ", dir=" << vy
            << ", dist=" << Dist << ", norm=" << normal << "\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0)); //&&ApproxEqual(normal,vy));
  Dist = Simple.DistanceToOut(Vec_t(0, -70, -1), vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0)); //&& ApproxEqual(normal, vmy));

  // SafetyToIn(P)

  Dist = Simple.SafetyToIn(pbigx);
  std::cout << "S2O unverified: Line " << __LINE__ << ", p=" << pbigx << ", safety=" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,20));
  Dist = Simple.SafetyToIn(pbigmx);
  std::cout << "S2O unverified: Line " << __LINE__ << ", p=" << pbigmx << ", safety=" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,20));
  Dist = Simple.SafetyToIn(pbigy);
  std::cout << "S2O unverified: Line " << __LINE__ << ", p=" << pbigmx << ", safety=" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,20));
  Dist = Simple.SafetyToIn(pbigmy);
  std::cout << "S2O unverified: Line " << __LINE__ << ", p=" << pbigmx << ", safety=" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,20));
  Dist = Simple.SafetyToIn(pbigz);
  std::cout << "S2O unverified: Line " << __LINE__ << ", p=" << pbigmx << ", safety=" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,80));
  Dist = Simple.SafetyToIn(pbigmz);
  std::cout << "S2O unverified: Line " << __LINE__ << ", p=" << pbigmx << ", safety=" << Dist << "\n";
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,80));

  // DistanceToIn(P,V)

  Dist = Simple.DistanceToIn(Vec_t(100, 0, -1), vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30));
  Dist = Simple.DistanceToIn(Vec_t(-100, 0, -1), vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30));
  Dist = Simple.DistanceToIn(Vec_t(0, 100, -5), vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30));
  Dist = Simple.DistanceToIn(Vec_t(0, -100, -5), vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30));
  Dist = Simple.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 90));
  Dist = Simple.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 90));
  Dist = Simple.DistanceToIn(pbigx, vxy);
  // std::cout <<"D2I unverified: Line "<< __LINE__ <<", p="<< pbigx <<", dir="<< vxy <<", dist="<<Dist<<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = Simple.DistanceToIn(pbigmx, vmxy);
  // std::cout <<"D2I unverified: Line "<< __LINE__ <<", p="<< pbigx <<", dir="<< vxy <<", dist="<<Dist<<"\n";
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  // Check Extent and cached BBox
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  Simple.Extent(minExtent, maxExtent);
  Simple.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-80, -80, -10)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(80, 80, 10)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));
  MyPCone->Extent(minExtent, maxExtent);
  MyPCone->GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
  // VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-80, -80, -20)));
  // VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(80, 80, 40)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));

#ifdef SCAN_SOLID

  std::cout << "\n=======     Polycone SCAN test      ========";
  std::cout << "\n\nPCone created ! " << std::endl;
  // -> Check methods :
  //  - Inside
  //  - DistanceToIn
  //  - DistanceToOut

  vecgeom::EnumInside in;

  std::cout << "\n\n==================================================";
  Vec_t pt(0, -100, 24);
  int y;
  for (y = -100; y <= 100; y += 10) {
    // pt.setY(y);
    pt.Set(0, y, 24);
    in = MyPCone->Inside(pt);

    std::cout << "\nx=" << pt.x() << "  y=" << pt.y() << "  z=" << pt.z();

    if (in == vecgeom::EInside::kInside)
      std::cout << " is inside";
    else if (in == vecgeom::EInside::kOutside)
      std::cout << " is outside";
    else
      std::cout << " is on the surface";
  }

  std::cout << "\n\n==================================================";
  Vec_t start(0, 0, -30);
  Vec_t dir(1. / std::sqrt(2.), 1. / std::sqrt(2.), 0), normal;
  Precision d;
  int z;

  std::cout << "\nPdep is (0, 0, z)";
  std::cout << "\nDir is (1, 1, 0)\n";

  for (z = -30; z <= 50; z += 5) {
    // start.setZ(z);
    start.Set(0, 0, z);

    in = MyPCone->Inside(start);
    std::cout << "x=" << start.x() << "  y=" << start.y() << "  z=" << start.z();

    if (in == vecgeom::EInside::kInside) {
      std::cout << " is inside";

      d = MyPCone->DistanceToOut(start, dir);
      std::cout << "  distance to out=" << d;
      d = MyPCone->SafetyToOut(start);
      std::cout << "  closest distance to out=" << d << std::endl;
    } else if (in == vecgeom::EInside::kOutside) {
      std::cout << " is outside";

      d = MyPCone->DistanceToIn(start, dir);
      std::cout << "  distance to in=" << d;
      d = MyPCone->SafetyToIn(start);
      std::cout << "  closest distance to in=" << d << std::endl;
    } else
      std::cout << " is on the surface" << std::endl;
  }

  std::cout << "\n\n==================================================";
  Vec_t start2(0, -100, -30);
  Vec_t dir2(0, 1, 0);
  Precision d2;

  std::cout << "\nPdep is (0, -100, z)";
  std::cout << "\nDir is (0, 1, 0)\n";

  for (z = -30; z <= 50; z += 5) {
    std::cout << "  z=" << z;
    // start2.setZ(z);
    start2.Set(0, -100, z);
    d2 = MyPCone->DistanceToIn(start2, dir2);
    std::cout << "  distance to in=" << d2;
    d2 = MyPCone->SafetyToIn(start2);
    std::cout << "  distance to in=" << d2 << std::endl;
  }

  std::cout << "\n\n==================================================";
  Vec_t start3(0, 0, -50);
  Vec_t dir3(0, 0, 1);
  Precision d3;

  std::cout << "\nPdep is (0, y, -50)";
  std::cout << "\nDir is (0, 0, 1)\n";

  for (y = -0; y <= 90; y += 5) {
    std::cout << "  y=" << y;
    // start3.setY(y);
    start3.Set(0, y, -50);
    d3 = MyPCone->DistanceToIn(start3, dir3);
    std::cout << "  distance to in=" << d3 << std::endl;
  }
  //
  // Add checks in Phi direction
  // Point move in Phi direction for differents Z
  //
  std::cout << "\n\n==================================================";
  Vec_t start4;
  for (z = -10; z <= 50; z += 5) {
    std::cout << "\n\n===================Z=" << z << "==============================";
    // Vec_t start4( 0, 0, z-0.00001);
    // Vec_t start4( 0, 0, z);
    start4.Set(0, 0, z);
    // G4double phi=pi/180.*rad;
    //  G4double phi=0.0000000001*pi/180.*rad;
    Precision phi = -kPi / 180. * kPi / 180.;
    Vec_t dir4(std::cos(phi), std::sin(phi), 0);
    Precision d4;

    std::cout << "\nPdep is (0<<R<<50, phi, z)";
    std::cout << "\nDir is (std::cos(phi), std::sin(phi), 0)\n";
    std::cout << "Ndirection is=" << dir4 << std::endl;

    for (y = -0; y <= 50; y += 5) {

      // start4.setX(y*std::cos(phi));
      // start4.setY(y*std::sin(phi));
      start4.Set(y * std::cos(phi), y * std::sin(phi), z);
      std::cout << "  R=" << y << " with Start" << start4;
      in = MyPCone->Inside(start4);
      if (in == vecgeom::EInside::kInside) {
        std::cout << " is inside";
        d4 = MyPCone->DistanceToOut(start4, dir4);
        std::cout << "  distance to out=" << d4;
        d4 = MyPCone->SafetyToOut(start4);
        std::cout << " closest distance to out=" << d4 << std::endl;
      } else if (in == vecgeom::EInside::kOutside) {
        std::cout << " is outside";
        d4 = MyPCone->DistanceToIn(start4, dir4);
        std::cout << "  distance to in=" << d4;
        d4 = MyPCone->SafetyToIn(start4);
        std::cout << " closest distance to in=" << d4 << std::endl;
      } else {
        std::cout << " is on the surface";
        d4 = MyPCone->DistanceToIn(start4, dir4);
        std::cout << "  distance to in=" << d4;
        d4 = MyPCone->SafetyToIn(start4);
        std::cout << " closest distance to in=" << d4 << std::endl;
      }
    }
  }
  //
  // Add checks in Phi direction
  // Point move in X direction for differents Z
  // and 'schoot' on rhi edge
  std::cout << "\n\n==================================================";
  Vec_t start5;
  for (z = -10; z <= 50; z += 5) {
    std::cout << "\n\n===================Z=" << z << "==============================";
    // Vec_t start5( 0., 0.000000000001, z);
    // Vec_t start5( 0., 1, z);
    start5.Set(0, 1, z);
    Vec_t dir5(0, -1, 0);
    Precision d5;

    std::cout << "\nPdep is (0<<X<<50, 1, z)";
    std::cout << "\nDir is (0, -1, 0)\n";
    std::cout << "Ndirection is=" << dir5 << std::endl;

    for (y = -0; y <= 50; y += 5) {

      // start5.setX(y);
      start5.Set(0, y, z);
      std::cout << " Start" << start5;
      in = MyPCone->Inside(start5);
      if (in == vecgeom::EInside::kInside) {
        std::cout << " is inside";
        d5 = MyPCone->DistanceToOut(start5, dir5);
        std::cout << "  distance to out=" << d5;
        d5 = MyPCone->SafetyToOut(start5);
        std::cout << " closest distance to out=" << d5 << std::endl;
      } else if (in == vecgeom::EInside::kOutside) {
        std::cout << " is outside";
        d5 = MyPCone->DistanceToIn(start5, dir5);
        std::cout << "  distance to in=" << d5;
        d5 = MyPCone->SafetyToIn(start5);
        std::cout << " closest distance to in=" << d5 << std::endl;
      } else {
        std::cout << " is on the surface";
        d5 = MyPCone->DistanceToIn(start5, dir5);
        std::cout << "  distance to in=" << d5;
        d5 = MyPCone->SafetyToIn(start5);
        std::cout << " closest distance to in=" << d5 << std::endl;
      }
    }
  }

#endif

  // Jira-175 test - old CMS volume SBSC
  const int Nz3        = 6;
  Precision rmin3[Nz3] = {53, 53, 53, 53, 114, 114};
  Precision rmax3[Nz3] = {54, 54, 115, 115, 115, 115};
  Precision z3[Nz3]    = {71, 121.5, 121.5, 122.5, 122.5, 282};
  Polycone_t pcon175("PCone175", 0, 360. * kPi / 180., Nz3, z3, rmin3, rmax3);

  Vec_t point175a{-18.1079855387881, -54.3917837284389, 121.5};

  VECGEOM_ASSERT(pcon175.Inside(point175a) == vecgeom::EInside::kSurface);

  Vec_t norm175;
  bool valid175 = pcon175.Normal(point175a, norm175);
  VECGEOM_ASSERT(ApproxEqual(norm175, Vec_t(0, 0, -1)) && valid175);

  {
    // Test cases corresponding to issue-618
    // Point is exactly on the joining disc between 2 sections of a polycone
    Precision rmin[]     = {0.1, 0.0, 0.0, 0.4};
    Precision rmax[]     = {1., 2., 2., 1.5};
    Precision z[]        = {-1, -0.5, 0.5, 2};
    int nZ               = 4;
    Polycone_t *newPCone = new Polycone_t("NewPCone", 0, 2 * kPi, nZ, z, rmin, rmax);
    Vec_t pin(0.4, 0., -0.5);
    VECGEOM_ASSERT(newPCone->SafetyToOut(pin) != 0.);
  }

  return true;
}

int main(int argc, char *argv[])
{
  TestPolycone<vecgeom::SimplePolycone>();
  std::cout << "VecGeomPolycone passed\n";
  return 0;
}
