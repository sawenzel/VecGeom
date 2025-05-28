// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for the Ellipsoid shape
/// @file test/unit_teststest/TestEllipsoid.cpp
/// @author Evgueni Tcherniaev

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/Ellipsoid.h"
#include "ApproxEqual.h"

bool testvecgeom = false;

using namespace vecgeom;

///////////////////////////////////////////////////////////////////////////////
//
// Estimate normal to the surface at given z and phi
//
Vector3D<Precision> EstimateNormal(Precision a, Precision b, Precision c, Precision z, Precision phi)
{
  Precision delta = 0.001;
  Precision z1    = z - delta;
  Precision z2    = z + delta;
  Precision phi1  = phi - delta;
  Precision phi2  = phi + delta;
  Precision rho1  = std::sqrt((1. + z1 / c) * (1. - z1 / c));
  Precision rho2  = std::sqrt((1. + z2 / c) * (1. - z2 / c));

  Vector3D<Precision> p1(rho1 * std::cos(phi1) * a, rho1 * std::sin(phi1) * b, z1);
  Vector3D<Precision> p2(rho1 * std::cos(phi2) * a, rho1 * std::sin(phi2) * b, z1);
  Vector3D<Precision> p3(rho2 * std::cos(phi1) * a, rho2 * std::sin(phi1) * b, z2);
  Vector3D<Precision> p4(rho2 * std::cos(phi2) * a, rho2 * std::sin(phi2) * b, z2);

  return ((p4 - p1).Cross(p3 - p2)).Unit();
}

///////////////////////////////////////////////////////////////////////////////
//
// Unit test for Ellipsoid
//
template <class Ellipsoid_t, class Vec_t = Vector3D<Precision>>
bool TestEllipsoid()
{
  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check surface area, volume and other basic methods
  //
  std::cout << "=== Check Set()/Get()" << std::endl;

  Precision a, b, c, zbottom, ztop;

  Ellipsoid_t solid("Test_Ellipsoid", a = 3., b = 4., c = 5., zbottom = -4.5, ztop = 3.5);
  VECGEOM_ASSERT(solid.GetDx() == a);
  VECGEOM_ASSERT(solid.GetDy() == b);
  VECGEOM_ASSERT(solid.GetDz() == c);
  VECGEOM_ASSERT(solid.GetZBottomCut() == zbottom);
  VECGEOM_ASSERT(solid.GetZTopCut() == ztop);

  solid.SetZCuts(-c - 1., c + 1);
  VECGEOM_ASSERT(solid.GetZBottomCut() == -c);
  VECGEOM_ASSERT(solid.GetZTopCut() == c);

  solid.SetZCuts(-4., 2);
  VECGEOM_ASSERT(solid.GetZBottomCut() == -4.);
  VECGEOM_ASSERT(solid.GetZTopCut() == 2.);

  solid.SetZCuts(0., 0.);
  VECGEOM_ASSERT(solid.GetZBottomCut() == -c);
  VECGEOM_ASSERT(solid.GetZTopCut() == c);

  solid.SetZCuts(zbottom, ztop);
  VECGEOM_ASSERT(solid.GetZBottomCut() == zbottom);
  VECGEOM_ASSERT(solid.GetZTopCut() == ztop);

  solid.SetSemiAxes(2., 3., 4.);
  VECGEOM_ASSERT(solid.GetDx() == 2.);
  VECGEOM_ASSERT(solid.GetDy() == 3.);
  VECGEOM_ASSERT(solid.GetDz() == 4.);
  VECGEOM_ASSERT(solid.GetZBottomCut() == -4.);
  VECGEOM_ASSERT(solid.GetZTopCut() == 3.5);

  solid.SetSemiAxes(a, b, c);
  solid.SetZCuts(zbottom, ztop);
  VECGEOM_ASSERT(solid.GetDx() == a);
  VECGEOM_ASSERT(solid.GetDy() == b);
  VECGEOM_ASSERT(solid.GetDz() == c);
  VECGEOM_ASSERT(solid.GetZBottomCut() == zbottom);
  VECGEOM_ASSERT(solid.GetZTopCut() == ztop);

  std::cout << "=== Check Print()" << std::endl;
  solid.Print();
  std::cout << "Ellipsoid (" << a << ", " << b << ", " << c << ", " << zbottom << ", " << ztop << ")" << std::endl;

  // Check Surface area
  int Npoints = 1000000;
  std::cout << "=== Check SurfaceArea()" << std::endl;

  // check sphere
  solid.SetSemiAxes(5., 5., 5.);
  solid.SetZCuts(0., 0.);
  Precision area      = solid.SurfaceArea();
  Precision areaMath  = 4. * vecgeom::kPi * 25.;
  Precision areaCheck = solid.GetUnplacedVolume()->EstimateSurfaceArea(Npoints);
  std::cout << " sphere(5) = " << area << "   exact = " << areaMath << "   mc_estimated = " << areaCheck << " ("
            << Npoints / 1000000. << " million points)" << std::endl;
  VECGEOM_ASSERT(std::abs(area - areaMath) < 0.01 * area);

  // check prolate spheroid
  solid.SetSemiAxes(3., 3., 5.);
  solid.SetZCuts(0., 0.);
  area        = solid.SurfaceArea();
  Precision e = 4. / 5.;
  areaMath    = vecgeom::kTwoPi * 3. * (3. + 5. * std::asin(e) / e);
  areaCheck   = solid.GetUnplacedVolume()->EstimateSurfaceArea(Npoints);
  std::cout << " spheroid(3,3,5) = " << area << "   exact = " << areaMath << "   mc_estimated = " << areaCheck << " ("
            << Npoints / 1000000. << " million points)" << std::endl;
  VECGEOM_ASSERT(std::abs(area - areaMath) < 0.01 * area);

  // check oblate spheroid
  solid.SetSemiAxes(5., 5., 3.);
  solid.SetZCuts(0., 0.);
  area      = solid.SurfaceArea();
  areaMath  = vecgeom::kTwoPi * 25. + vecgeom::kPi * 9. * std::log((1. + e) / (1 - e)) / e;
  areaCheck = solid.GetUnplacedVolume()->EstimateSurfaceArea(Npoints);
  std::cout << " spheroid(5,5,3) = " << area << "   exact = " << areaMath << "   mc_estimated = " << areaCheck << " ("
            << Npoints / 1000000. << " million points)" << std::endl;
  VECGEOM_ASSERT(std::abs(area - areaMath) < 0.01 * area);

  // check ellipsoid under test
  solid.SetSemiAxes(a, b, c);
  solid.SetZCuts(zbottom, ztop);
  area      = solid.SurfaceArea();
  areaCheck = solid.GetUnplacedVolume()->EstimateSurfaceArea(Npoints);
  std::cout << " ellipsoid(3,4,5, -4.5,3.5) = " << area << "   mc_estimated = " << areaCheck << " ("
            << Npoints / 1000000. << " million points)" << std::endl;
  VECGEOM_ASSERT(std::abs(area - areaCheck) < 0.01 * area);

  // Check Cubic volume
  std::cout << "=== Check Capacity()" << std::endl;
  solid.SetSemiAxes(a, b, c);
  solid.SetZCuts(zbottom, ztop);
  Precision vol      = solid.Capacity();
  Precision volCheck = solid.GetUnplacedVolume()->EstimateCapacity(Npoints);
  std::cout << " volume = " << vol << "   mc_estimated = " << volCheck << " (" << Npoints / 1000000.
            << " million points)" << std::endl;
  VECGEOM_ASSERT(std::abs(vol - volCheck) < 0.01 * vol);

  // Check Extent
  std::cout << "=== Check Extent()" << std::endl;
  Vec_t minCheck(kInfLength, kInfLength, kInfLength);
  Vec_t maxCheck(-kInfLength, -kInfLength, -kInfLength);
  for (int i = 0; i < Npoints; ++i) {
    Vec_t p = solid.GetUnplacedVolume()->SamplePointOnSurface();
    minCheck.Set(std::min(p.x(), minCheck.x()), std::min(p.y(), minCheck.y()), std::min(p.z(), minCheck.z()));
    maxCheck.Set(std::max(p.x(), maxCheck.x()), std::max(p.y(), maxCheck.y()), std::max(p.z(), maxCheck.z()));
  }
  Vec_t minExtent, maxExtent;
  solid.Extent(minExtent, maxExtent);
  std::cout << " calculated:    min = " << minExtent << " max = " << maxExtent << std::endl;
  std::cout << " mc_estimated:  min = " << minCheck << " max = " << maxCheck << " (" << Npoints / 1000000.
            << " million points)" << std::endl;

  VECGEOM_ASSERT(std::abs(minExtent.x() - minCheck.x()) < 0.001 * std::abs(minExtent.x()));
  VECGEOM_ASSERT(std::abs(minExtent.y() - minCheck.y()) < 0.001 * std::abs(minExtent.y()));
  VECGEOM_ASSERT(minExtent.z() == minCheck.z());
  VECGEOM_ASSERT(std::abs(maxExtent.x() - maxCheck.x()) < 0.001 * std::abs(maxExtent.x()));
  VECGEOM_ASSERT(std::abs(maxExtent.y() - maxCheck.y()) < 0.001 * std::abs(maxExtent.y()));
  VECGEOM_ASSERT(maxExtent.z() == maxCheck.z());

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check Inside()
  //
  std::cout << "=== Check Inside()" << std::endl;
  solid.SetSemiAxes(a, b, c);
  solid.SetZCuts(zbottom, ztop);
  int NZ         = 30;
  int NPHI       = 30;
  Precision DZ   = 2. * c / NZ;
  Precision DPHI = kTwoPi / NPHI;
  for (int iz = 0; iz < NZ; ++iz) {
    Precision z = -c + iz * DZ;
    for (int iphi = 0; iphi < NPHI; ++iphi) {
      Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
      Precision phi = iphi * DPHI;
      Precision rho = std::sqrt((1. + z / c) * (1. - z / c));
      Precision px  = rho * std::cos(phi) * a + eps;
      Precision py  = rho * std::sin(phi) * b + eps;
      Precision pz  = z + eps;
      if (z < zbottom) pz = zbottom;
      if (z > ztop) pz = ztop;
      Vec_t p(px, py, pz);
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kSurface);
      VECGEOM_ASSERT(solid.Inside(p * 0.999) == vecgeom::kInside);
      VECGEOM_ASSERT(solid.Inside(p * 1.001) == vecgeom::kOutside);
    }
  }
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., zbottom)) == vecgeom::kSurface);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., zbottom - kTolerance)) == vecgeom::kOutside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., zbottom + kTolerance)) == vecgeom::kInside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., ztop)) == vecgeom::kSurface);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., ztop - kTolerance)) == vecgeom::kInside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., ztop + kTolerance)) == vecgeom::kOutside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., 0.)) == vecgeom::kInside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., -c)) == vecgeom::kOutside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., 0., +c)) == vecgeom::kOutside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(-a, 0., 0.)) == vecgeom::kSurface);
  VECGEOM_ASSERT(solid.Inside(Vec_t(-a - kTolerance, 0., 0.)) == vecgeom::kOutside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(-a + kTolerance, 0., 0.)) == vecgeom::kInside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., b, 0.)) == vecgeom::kSurface);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., b - kTolerance, 0.)) == vecgeom::kInside);
  VECGEOM_ASSERT(solid.Inside(Vec_t(0., b + kTolerance, 0.)) == vecgeom::kOutside);

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check Normal()
  //
  std::cout << "=== Check Normal()" << std::endl;
  solid.SetSemiAxes(a, b, c);
  solid.SetZCuts(zbottom, ztop);
  Vec_t normal(0.);
  bool valid;

  // Check normals on lateral surface
  NZ   = 30;
  NPHI = 30;
  DZ   = (ztop - zbottom) / NZ;
  DPHI = kTwoPi / NPHI;
  for (int iz = 1; iz < NZ - 1; ++iz) {
    Precision z = zbottom + iz * DZ;
    for (int iphi = 0; iphi < NPHI; ++iphi) {
      Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
      Precision phi = iphi * DPHI;
      Precision rho = std::sqrt((1. + z / c) * (1. - z / c));
      Precision px  = rho * std::cos(phi) * a + eps;
      Precision py  = rho * std::sin(phi) * b + eps;
      Precision pz  = z + eps;
      valid         = solid.Normal(Vec_t(px, py, pz), normal);
      VECGEOM_ASSERT(valid);
      VECGEOM_ASSERT(ApproxEqual(normal, EstimateNormal(a, b, c, pz, phi)));
    }
  }

  // Check normals at zbottom edge
  for (int iphi = 0; iphi < NPHI; ++iphi) {
    Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
    Precision phi = iphi * DPHI;
    Precision rho = std::sqrt((1. + zbottom / c) * (1. - zbottom / c));
    Precision px  = rho * std::cos(phi) * a + eps;
    Precision py  = rho * std::sin(phi) * b + eps;
    Precision pz  = zbottom + eps;
    valid         = solid.Normal(Vec_t(px, py, pz), normal);
    VECGEOM_ASSERT(valid);
    VECGEOM_ASSERT(ApproxEqual(normal, (EstimateNormal(a, b, c, pz, phi) + Vec_t(0., 0., -1.)).Unit()));
  }

  // Check normals at ztop edge
  for (int iphi = 0; iphi < NPHI; ++iphi) {
    Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
    Precision phi = iphi * DPHI;
    Precision rho = std::sqrt((1. + ztop / c) * (1. - ztop / c));
    Precision px  = rho * std::cos(phi) * a + eps;
    Precision py  = rho * std::sin(phi) * b + eps;
    Precision pz  = ztop + eps;
    valid         = solid.Normal(Vec_t(px, py, pz), normal);
    VECGEOM_ASSERT(valid);
    VECGEOM_ASSERT(ApproxEqual(normal, (EstimateNormal(a, b, c, pz, phi) + Vec_t(0., 0., 1.)).Unit()));
  }

  // Check normals on zbottom cut
  VECGEOM_ASSERT(solid.Normal(Vec_t(0., 0., zbottom), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(0.5, 0., zbottom), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(0, -0.5, zbottom), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(-0.5, 0.5, zbottom), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(-0.6, -0.6, zbottom), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));

  // Check normals on ztop cut
  VECGEOM_ASSERT(solid.Normal(Vec_t(0., 0., ztop), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(0.5, 0., ztop), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(0, -0.5, ztop), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(-0.5, 0.5, ztop), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));
  VECGEOM_ASSERT(solid.Normal(Vec_t(-0.6, -0.6, ztop), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));

  // Check points not on surface
  VECGEOM_ASSERT(!solid.Normal(Vec_t(0., 0., 0.), normal));
  VECGEOM_ASSERT(normal.Mag() == 1.);

  VECGEOM_ASSERT(!solid.Normal(Vec_t(0., 0., zbottom + 1.), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));
  VECGEOM_ASSERT(!solid.Normal(Vec_t(0.1, 0.1, zbottom + 1.), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., -1.));

  VECGEOM_ASSERT(!solid.Normal(Vec_t(0., 0., ztop - 1.), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));
  VECGEOM_ASSERT(!solid.Normal(Vec_t(-0.2, -0.3, ztop - 1.), normal));
  VECGEOM_ASSERT(normal == Vec_t(0., 0., 1.));

  for (int iz = 1; iz < NZ - 1; ++iz) {
    Precision z = zbottom + iz * DZ;
    for (int iphi = 0; iphi < NPHI; ++iphi) {
      Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
      Precision phi = iphi * DPHI;
      Precision rho = std::sqrt((1. + z / c) * (1. - z / c));
      Precision px  = rho * std::cos(phi) * a + eps;
      Precision py  = rho * std::sin(phi) * b + eps;
      Precision pz  = z + eps;
      valid         = solid.Normal(Vec_t(px, py, pz) * 1.2, normal);
      VECGEOM_ASSERT(!valid);
      VECGEOM_ASSERT(ApproxEqual(normal, EstimateNormal(a, b, c, pz, phi)));
      valid = solid.Normal(Vec_t(px, py, pz) * 0.8, normal);
      VECGEOM_ASSERT(!valid);
      VECGEOM_ASSERT(ApproxEqual(normal, EstimateNormal(a, b, c, pz, phi)));
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check SafetyToIn()
  //
  std::cout << "=== Check SafetyToIn()" << std::endl;
  solid.SetSemiAxes(a = 3, b = 4, c = 5);
  solid.SetZCuts(zbottom = -4.5, ztop = 3.5);

  // Check consistence with the convention
  NZ   = 30;
  NPHI = 30;
  DZ   = 2. * c / NZ;
  DPHI = kTwoPi / NPHI;
  for (int iz = 0; iz < NZ; ++iz) {
    Precision z = -c + iz * DZ;
    for (int iphi = 0; iphi < NPHI; ++iphi) {
      Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
      Precision phi = iphi * DPHI;
      Precision rho = std::sqrt((1. + z / c) * (1. - z / c));
      Precision px  = rho * std::cos(phi) * a + eps;
      Precision py  = rho * std::sin(phi) * b + eps;
      Precision pz  = z + eps;
      if (z < zbottom) pz = zbottom;
      if (z > ztop) pz = ztop;
      Vec_t p(px, py, pz);
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kSurface);
      VECGEOM_ASSERT(solid.SafetyToIn(p) == 0.);
      p = Vec_t(px, py, pz) * 2.;
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kOutside);
      VECGEOM_ASSERT(solid.SafetyToIn(p) > 0.);
      p = Vec_t(px, py, pz) * 0.5;
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kInside);
      VECGEOM_ASSERT(solid.SafetyToIn(p) < 0.);
    }
  }

  // Check particular points to verify that the algorithm works as expected
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(+10, 0, 0)) == 10. - a);
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(-10, 0, 0)) == 10. - a);
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(0, +10, 0)) == 10. - b);
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(0, -10, 0)) == 10. - b);
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(0, 0, +10)) == 10. - ztop);
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(0, 0, -10)) == 10. + zbottom);
  VECGEOM_ASSERT(solid.SafetyToIn(Vec_t(a, b, ztop)) > 0.);
  VECGEOM_ASSERT(ApproxEqual<Precision>(solid.SafetyToIn(Vec_t(a, b, c)), (std::sqrt(3.) - 1.) * a));

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check SafetyToOut()
  //
  std::cout << "=== Check SafetyToOut()" << std::endl;
  solid.SetSemiAxes(a = 3, b = 4, c = 5);
  solid.SetZCuts(zbottom = -4.5, ztop = 3.5);

  // Check consistence with the convention
  NZ   = 30;
  NPHI = 30;
  DZ   = 2. * c / NZ;
  DPHI = kTwoPi / NPHI;
  for (int iz = 0; iz < NZ; ++iz) {
    Precision z = -c + iz * DZ;
    for (int iphi = 0; iphi < NPHI; ++iphi) {
      Precision eps = 0.5 * kHalfTolerance * (2. * RNG::Instance().uniform() - 1.);
      Precision phi = iphi * DPHI;
      Precision rho = std::sqrt((1. + z / c) * (1. - z / c));
      Precision px  = rho * std::cos(phi) * a + eps;
      Precision py  = rho * std::sin(phi) * b + eps;
      Precision pz  = z + eps;
      if (z < zbottom) pz = zbottom;
      if (z > ztop) pz = ztop;
      Vec_t p(px, py, pz);
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kSurface);
      VECGEOM_ASSERT(solid.SafetyToOut(p) == 0.);
      p = Vec_t(px, py, pz) * 2.;
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kOutside);
      VECGEOM_ASSERT(solid.SafetyToOut(p) < 0.);
      p = Vec_t(px, py, pz) * 0.5;
      VECGEOM_ASSERT(solid.Inside(p) == vecgeom::kInside);
      VECGEOM_ASSERT(solid.SafetyToOut(p) > 0.);
    }
  }

  // Check particular points to verify that the algorithm works as expected
  VECGEOM_ASSERT(solid.SafetyToOut(Vec_t(0, 0, 0)) == a);
  VECGEOM_ASSERT(solid.SafetyToOut(Vec_t(0, 0, 2.5)) == ztop - 2.5);

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check DistanceToIn()
  //
  std::cout << "=== Check DistanceToIn()" << std::endl;
  solid.SetSemiAxes(a = 3, b = 4, c = 5);
  solid.SetZCuts(zbottom = -4.5, ztop = 3.5);
  Precision sctop = std::sqrt((c - ztop) * (2. - c + ztop));
  Precision scbot = std::sqrt((c + zbottom) * (2. - c - zbottom));
  Precision del   = kTolerance / 3.;

  // set coordinates for points in grid
  static const int np = 11;

  Precision xxx[np] = {-a - 1, -a - del, -a, -a + del, -1, 0, 1, a - del, a, a + del, a + 1};
  Precision yyy[np] = {-b - 1, -b - del, -b, -b + del, -1, 0, 1, b - del, b, b + del, b + 1};
  Precision zzz[np] = {-4.5 - 1, -4.5f - del, -4.5, -4.5f + del, -1, 0, 1, 3.5f - del, 3.5, 3.5f + del, 3.5 + 1};

  // check directions parallel to +Z
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToIn(p, Vec_t(0, 0, 1));
        // Check inside points ("wrong" side)
        if (solid.Inside(p) == vecgeom::kInside) VECGEOM_ASSERT(dist < 0);
        // Check points on surface
        if (solid.Inside(p) == vecgeom::kSurface) {
          if (p.z() > zbottom + 1) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            VECGEOM_ASSERT(ApproxEqual<Precision>(dist, zbottom - p.z()));
          }
        }
        // Check outside points
        if (solid.Inside(p) == vecgeom::kOutside) {
          if (p.z() >= 0 || (p.x() * p.x() / (a * a) + p.y() * p.y() / (b * b)) >= 1.) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            if (p.x() * p.x() / (a * a * scbot * scbot) + p.y() * p.y() / (b * b * scbot * scbot) <= 1.) {
              VECGEOM_ASSERT(ApproxEqual<Precision>(dist, zbottom - p.z()));
            } else {
              Precision z = c * std::sqrt(1. - p.x() * p.x() / (a * a) - p.y() * p.y() / (b * b));
              VECGEOM_ASSERT(ApproxEqual<Precision>(dist, -z - p.z()));
            }
          }
        }
      }
    }
  }

  // check directions parallel to -Z
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToIn(p, Vec_t(0, 0, -1));
        // Check inside points ("wrong" side)
        if (solid.Inside(p) == vecgeom::kInside) VECGEOM_ASSERT(dist < 0);
        // Check points on surface
        if (solid.Inside(p) == vecgeom::kSurface) {
          if (p.z() < ztop - 1) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            VECGEOM_ASSERT(ApproxEqual<Precision>(dist, p.z() - ztop));
          }
        }
        // Check outside points
        if (solid.Inside(p) == vecgeom::kOutside) {
          if (p.z() <= 0 || (p.x() * p.x() / (a * a) + p.y() * p.y() / (b * b)) >= 1.) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            if (p.x() * p.x() / (a * a * sctop * sctop) + p.y() * p.y() / (b * b * sctop * sctop) <= 1.) {
              VECGEOM_ASSERT(ApproxEqual<Precision>(dist, p.z() - ztop));
            } else {
              Precision z = c * std::sqrt(1. - p.x() * p.x() / (a * a) - p.y() * p.y() / (b * b));
              VECGEOM_ASSERT(ApproxEqual<Precision>(dist, p.z() - z));
            }
          }
        }
      }
    }
  }

  // check directions parallel to +Y
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToIn(p, Vec_t(0, 1, 0));
        if (solid.Inside(p) == vecgeom::kInside) {
          VECGEOM_ASSERT(dist < 0);
        } else {
          if (p.y() >= 0 || p.z() < zbottom + kHalfTolerance || p.z() > ztop - kHalfTolerance ||
              (p.x() * p.x() / (a * a) + p.z() * p.z() / (c * c)) >= 1.) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            Precision y = b * std::sqrt(1. - p.x() * p.x() / (a * a) - p.z() * p.z() / (c * c));
            VECGEOM_ASSERT(ApproxEqual<Precision>(dist, -y - p.y()));
          }
        }
      }
    }
  }

  // check directions parallel to -X
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToIn(p, Vec_t(-1, 0, 0));
        if (solid.Inside(p) == vecgeom::kInside) {
          VECGEOM_ASSERT(dist < 0);
        } else {
          if (p.x() <= 0 || p.z() < zbottom + kHalfTolerance || p.z() > ztop - kHalfTolerance ||
              (p.y() * p.y() / (b * b) + p.z() * p.z() / (c * c)) >= 1.) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            Precision x = a * std::sqrt(1. - p.y() * p.y() / (b * b) - p.z() * p.z() / (c * c));
            VECGEOM_ASSERT(ApproxEqual<Precision>(dist, p.x() - x));
          }
        }
      }
    }
  }

  // check far points
  Precision Kfar = 1.e+5;
  int nz         = 40;
  int nphi       = 36;
  for (int iz = 0; iz < nz + 1; ++iz) {
    for (int iphi = 0; iphi < nphi; ++iphi) {
      Precision z   = -1. + iz * (2. / nz);
      Precision phi = iphi * (kTwoPi / nphi);
      Precision rho = std::sqrt(1. - z * z);
      Precision x   = rho * std::cos(phi);
      Precision y   = rho * std::sin(phi);
      if (z < zbottom / c) {
        x = a * x * zbottom / (c * z);
        y = b * y * zbottom / (c * z);
        z = zbottom;
      } else if (z > ztop / c) {
        x = a * x * ztop / (c * z);
        y = b * y * ztop / (c * z);
        z = ztop;
      } else {
        x *= a;
        y *= b;
        z *= c;
      }
      Vec_t p(x, y, z);
      Vec_t v        = -p.Unit();
      Precision dist = solid.DistanceToIn(Kfar * p, v);
      VECGEOM_ASSERT(std::abs(dist - (Kfar - 1.) * p.Mag()) < kHalfTolerance);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check DistanceToOut()
  //
  std::cout << "=== Check DistanceToOut()" << std::endl;
  solid.SetSemiAxes(a = 3, b = 4, c = 5);
  solid.SetZCuts(zbottom = -4.5, ztop = 3.5);

  // check directions parallel to +Z
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToOut(p, Vec_t(0, 0, 1));
        // Check if point is outside ("wrong" side)
        if (solid.Inside(p) == vecgeom::kOutside) {
          VECGEOM_ASSERT(dist < 0);
        } else {
          if ((p.x() * p.x() / (a * a) + p.y() * p.y() / (b * b)) >= 1.) {
            VECGEOM_ASSERT(dist == 0.);
          } else if (solid.Inside(Vec_t(p.x(), p.y(), ztop)) == vecgeom::kSurface) {
            VECGEOM_ASSERT(std::abs(dist - (ztop - p.z())) < 0.01 * kHalfTolerance);
          } else {
            Precision z = c * std::sqrt(1. - p.x() * p.x() / (a * a) - p.y() * p.y() / (b * b));
            VECGEOM_ASSERT(std::abs(dist - (z - p.z())) < 0.01 * kHalfTolerance);
          }
        }
      }
    }
  }

  // check directions parallel to -Z
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToOut(p, Vec_t(0, 0, -1));
        // Check if point is outside ("wrong" side)
        if (solid.Inside(p) == vecgeom::kOutside) {
          VECGEOM_ASSERT(dist < 0);
        } else {
          if ((p.x() * p.x() / (a * a) + p.y() * p.y() / (b * b)) >= 1.) {
            VECGEOM_ASSERT(dist == 0.);
          } else if (solid.Inside(Vec_t(p.x(), p.y(), zbottom)) == vecgeom::kSurface) {
            VECGEOM_ASSERT(std::abs(dist - (p.z() - zbottom)) < 0.01 * kHalfTolerance);
          } else {
            Precision z = c * std::sqrt(1. - p.x() * p.x() / (a * a) - p.y() * p.y() / (b * b));
            VECGEOM_ASSERT(std::abs(dist - (p.z() + z)) < 0.01 * kHalfTolerance);
          }
        }
      }
    }
  }

  // check directions parallel to -Y
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToOut(p, Vec_t(0, -1, 0));
        // Check if point is outside ("wrong" side)
        if (solid.Inside(p) == vecgeom::kOutside) {
          VECGEOM_ASSERT(dist < 0);
        } else {
          if ((p.x() * p.x() / (a * a) + p.z() * p.z() / (c * c)) >= 1.) {
            VECGEOM_ASSERT(dist == 0.);
          } else {
            Precision y = b * std::sqrt(1. - p.x() * p.x() / (a * a) - p.z() * p.z() / (c * c));
            VECGEOM_ASSERT(std::abs(dist - (p.y() + y)) < 0.01 * kHalfTolerance);
          }
        }
      }
    }
  }

  // check directions parallel to +X
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        Vec_t p(xxx[ix], yyy[iy], zzz[iz]);
        Precision dist = solid.DistanceToOut(p, Vec_t(1, 0, 0));
        // Check if point is outside ("wrong" side)
        if (solid.Inside(p) == vecgeom::kOutside) {
          VECGEOM_ASSERT(dist < 0);
        } else {
          if ((p.y() * p.y() / (b * b) + p.z() * p.z() / (c * c)) >= 1.) {
            VECGEOM_ASSERT(dist == 0.);
          } else {
            Precision x = a * std::sqrt(1. - p.y() * p.y() / (b * b) - p.z() * p.z() / (c * c));
            VECGEOM_ASSERT(std::abs(dist - (x - p.x())) < 0.01 * kHalfTolerance);
          }
        }
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check SamplePointOnSurface()
  //
  std::cout << "=== Check SamplePointOnSurface()" << std::endl;
  solid.SetSemiAxes(a = 3, b = 4, c = 5);
  solid.SetZCuts(zbottom = -4.5, ztop = 3.5);
  area            = solid.SurfaceArea();
  Precision hbot  = 1. + zbottom / c;
  Precision htop  = 1. - ztop / c;
  Precision szneg = kPi * a * b * hbot * (2. - hbot);
  Precision szpos = kPi * a * b * htop * (2. - htop);
  Precision sside = area - szneg - szpos;

  Stopwatch timer;
  timer.Start();
  int nzneg = 0, nzpos = 0, nside = 0, nfactor = 100000, ntot = area * nfactor;
  for (int i = 0; i < ntot; i++) {
    Vec_t rndPoint = solid.GetUnplacedVolume()->SamplePointOnSurface();
    VECGEOM_ASSERT(solid.Inside(rndPoint) == vecgeom::kSurface);
    if (rndPoint.z() == zbottom)
      ++nzneg;
    else if (rndPoint.z() == ztop)
      ++nzpos;
    else
      ++nside;
  }
  timer.Stop();
  std::cout << "szneg,sside,szpos = " << szneg << ", \t" << sside << ", \t" << szpos << std::endl;
  std::cout << "nzneg,nside,nzpos = " << nzneg << ", \t" << nside << ", \t" << nzpos << std::endl;
  VECGEOM_ASSERT(std::abs(nzneg - szneg * nfactor) < 2. * std::sqrt(ntot));
  VECGEOM_ASSERT(std::abs(nside - sside * nfactor) < 2. * std::sqrt(ntot));
  VECGEOM_ASSERT(std::abs(nzpos - szpos * nfactor) < 2. * std::sqrt(ntot));
  std::cout << "Time : " << timer.Elapsed() << " sec   " << ntot / 1000000. << " million points" << std::endl;
  std::cout << "Time per million points : " << timer.Elapsed() * 1000000. / ntot << " sec" << std::endl;

  return true;
}

int main(int argc, char *argv[])
{
  VECGEOM_ASSERT(TestEllipsoid<vecgeom::SimpleEllipsoid>());
  std::cout << "VecGeom Ellipsoid passed\n";

  return 0;
}
