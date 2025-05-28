/*
 * TestVecGeomPolycone.cpp
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

// a file to test compilation of features during development

//.. ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "ApproxEqual.h"
#include "VecGeom/management/GeoManager.h"

using Real_v = vecgeom::VectorBackend::Real_v;

#include <iostream>

using namespace vecgeom;

bool testingvecgeom = false;

typedef Vector3D<Precision> Vec3D_t;

template <class Polycone_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>

bool TestPolycone()
{
  return 0;
}

int main()
{
  int Nz = 4;
  // a few cones
  Precision rmin[] = {0.1, 0.0, 0.0, 0.4};
  Precision rmax[] = {1., 2., 2., 1.5};
  Precision z[]    = {-1, -0.5, 0.5, 2};

  auto poly1 = GeoManager::MakeInstance<UnplacedPolycone>(0.,      /* initial phi starting angle */
                                                          kTwoPi,  /* total phi angle */
                                                          Nz,      /* number corners in r,z space */
                                                          z, rmin, /* r coordinate of these corners */
                                                          rmax);

  // poly1.Print();

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

  // test bounding box
  Vector3D<Precision> minExtent, maxExtent;
  Vector3D<Precision> minBBox, maxBBox;
  poly1->Extent(minExtent, maxExtent);
  poly1->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual<Precision>(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual<Precision>(maxExtent, maxBBox));

  // create a place version
  VPlacedVolume const *placedpoly1 = (new LogicalVolume("poly1", poly1))->Place(new Transformation3D());

  // test contains/inside
  VECGEOM_ASSERT(placedpoly1->Contains(Vec3D_t(0., 0., 0.)) == true);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec3D_t(0., 0., -2.)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec3D_t(0., 0., -0.8)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec3D_t(0., 0., -1.8)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec3D_t(0., 0., 10)) == false);
  VECGEOM_ASSERT(placedpoly1->Contains(Vec3D_t(0., 0., 1.8)) == false);

  // test DistanceToIn
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToIn(Vec3D_t(0., 0., -3.), Vec3D_t(0., 0., 1.)), 2.5));
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec3D_t(0., 0., -2.), Vec3D_t(0., 0., -1.)) == vecgeom::kInfLength);
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToIn(Vec3D_t(0., 0., 3), Vec3D_t(0., 0., -1.)), 2.5));
  VECGEOM_ASSERT(placedpoly1->DistanceToIn(Vec3D_t(0., 0., 3), Vec3D_t(0., 0., 1.)) == vecgeom::kInfLength);
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToIn(Vec3D_t(3., 0., 0), Vec3D_t(-1., 0., 0.)), 1.));
  VECGEOM_ASSERT(
      ApproxEqual<Precision>(placedpoly1->DistanceToIn(Vec3D_t(0., 0., 1.9999999), Vec3D_t(1., 0., 0.)), 0.4));

  // test SafetyToIn
  // VECGEOM_ASSERT( placedpoly1-> SafetyToIn( Vec3D_t(0.,0.,-3.)) == 2. );
  // VECGEOM_ASSERT( placedpoly1-> SafetyToIn( Vec3D_t(0.5,0.,-1.)) == 0. );
  // VECGEOM_ASSERT( placedpoly1-> SafetyToIn( Vec3D_t(0.,0.,3) ) == 1 );
  // VECGEOM_ASSERT( placedpoly1-> SafetyToIn( Vec3D_t(2.,0.,0.1) ) == 0 );
  // VECGEOM_ASSERT( placedpoly1-> SafetyToIn( Vec3D_t(3.,0.,0) ) == 1. );
  // std::cout << "test SIN " << placedpoly1->SafetyToIn(Vec3D_t(8., 0., 0.)) << std::endl;
  // test SafetyToOut
  // std::cout << "test SOUT " << placedpoly1->SafetyToOut(Vec3D_t(0., 0., 0.)) << std::endl;
  // VECGEOM_ASSERT( placedpoly1-> SafetyToOut( Vec3D_t(0.,0.,0.)) == 0.5 );
  //  VECGEOM_ASSERT( placedpoly1-> SafetyToOut( Vec3D_t(0.,0.,0.5)) == 0. );
  //  VECGEOM_ASSERT( placedpoly1-> SafetyToOut( Vec3D_t(1.9,0.,0) ) == 0.1 );
  // VECGEOM_ASSERT( placedpoly1-> SafetyToOut( Vec3D_t(0.2,0.,-1) ) == 0. );
  //  VECGEOM_ASSERT( placedpoly1-> SafetyToOut( Vec3D_t(1.4,0.,2) ) == 0. );

  // test DistanceToOut
  // [see VECGEOM-414] VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(0., 0., 0.), Vec3D_t(0.,
  // 0., 1.)), 0.5));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(0., 0., 0.), Vec3D_t(0., 0., -1.)), 0.5));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(2., 0., 0.), Vec3D_t(1., 0., 0.)), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(2., 0., 0.), Vec3D_t(-1., 0., 0.)), 4.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(1., 0., 2), Vec3D_t(0., 0., 1.)), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(0.5, 0., -1), Vec3D_t(0., 0., -1.)), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedpoly1->DistanceToOut(Vec3D_t(0.5, 0., -1), Vec3D_t(0., 0., 1.)), 3.));
  std::cout << "VecGeomPolycone tests passed.\n";
  return 0;
}
