//
// File:    TestPolyhedra
// Purpose: Polyhedra unit tests
//

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "ApproxEqual.h"

#include <cmath>

//-- ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"
#include <cassert>

using vecgeom::kPi;

template <class Polyhedra_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestPolyhedra()
{
  Precision tolerance = vecgeom::kTolerance;

  Precision RMINVec[8];
  RMINVec[0] = 30;
  RMINVec[1] = 30;
  RMINVec[2] = 0;
  RMINVec[3] = 0;
  RMINVec[4] = 0;
  RMINVec[5] = 0;
  RMINVec[6] = 40;
  RMINVec[7] = 40;

  Precision RMAXVec[8];
  RMAXVec[0] = 70;
  RMAXVec[1] = 70;
  RMAXVec[2] = 70;
  RMAXVec[3] = 40;
  RMAXVec[4] = 40;
  RMAXVec[5] = 80;
  RMAXVec[6] = 80;
  RMAXVec[7] = 60;

  Precision Z_Values[8];
  Z_Values[0] = -30;
  Z_Values[1] = -20;
  Z_Values[2] = -10;
  Z_Values[3] = 0;
  Z_Values[4] = 10;
  Z_Values[5] = 20;
  Z_Values[6] = 30;
  Z_Values[7] = 40;

  Precision sphi      = 0.0;
  Precision dphi      = kPi / 4.;
  Precision halfdphi  = 0.5 * dphi / 5.;
  Polyhedra_t *MyPGon = new Polyhedra_t("MyPGon", sphi, dphi, 5, 8, Z_Values, RMINVec, RMAXVec);

  Precision RMINVec0[2];
  RMINVec0[0] = 1;
  RMINVec0[1] = 1;

  Precision RMAXVec0[2];
  RMAXVec0[0] = 2;
  RMAXVec0[1] = 2;

  Precision Z_Values0[2];
  Z_Values0[0] = -1;
  Z_Values0[1] = 1;

  Precision sphi0 = 0.0;
  Precision dphi0 = kPi;

  Polyhedra_t *MyPGon0 = new Polyhedra_t("MyPGon0", sphi0, dphi0, 2, 2, Z_Values0, RMINVec0, RMAXVec0);

  Precision RMINVec1[3];
  RMINVec1[0] = 0;
  RMINVec1[1] = 0;
  RMINVec1[2] = 0;

  Precision RMAXVec1[3];
  RMAXVec1[0] = 2;
  RMAXVec1[1] = 1;
  RMAXVec1[2] = 2;

  Precision Z_Values1[3];
  Z_Values1[0] = -1;
  Z_Values1[1] = 0;
  Z_Values1[2] = 1;

  Precision sphi1 = 0.;
  Precision dphi1 = 2 * kPi;

  Polyhedra_t *MyPGon1 = new Polyhedra_t("MyPGon1", sphi1, dphi1, 4, 3, Z_Values1, RMINVec1, RMAXVec1);

  const int Nrz = 4, Nside = 6;
  Precision zz[Nrz] = {10, -10, -10, 10};
  Precision rr[Nrz] = {15, 15, 0, 0};

  vecgeom::SimplePolyhedron *MyPGon2 =
      new vecgeom::SimplePolyhedron("Hexagonal prism", sphi1, dphi1, Nside, Nrz, rr, zz);

  // std::cout << "=== Polyhedron: \n";
  // std::cout << *MyPGon2 << std::endl;

  // Check Cubic volume
  // Precision vol;
  // vol = MyPGon->Capacity();
  std::cout.precision(20);
  std::cout << "Complex Polyhedron Capacity =" << MyPGon->Capacity() << std::endl;
  // assert(ApproxEqual(vol,155138.6874225));

  // Check Surface area
  // vol=MyPGon->SurfaceArea();
  // assert(ApproxEqual(vol,1284298.5697));
  std::cout << "Complex Polyhedron SurfaceArea =" << MyPGon->SurfaceArea() << std::endl;

  // Check Cubic volume

  // vol = MyPGon0->Capacity();
  std::cout.precision(20);
  std::cout << "Simple Polyhedron(HalfBox) Capacity =" << MyPGon0->Capacity() << " has to be 12" << std::endl;
  std::cout << "Less Simple Polyhedron(2 cutted piramides) Capacity =" << MyPGon1->Capacity() << " has to be 18, ..."
            << std::endl;
  // assert(ApproxEqual(vol,155138.6874225));

  // Check Surface area
  // vol=MyPGon0->SurfaceArea();
  // assert(ApproxEqual(vol,1284298.5697));
  std::cout << "Simple Polyhedron(Half) SurfaceArea =" << MyPGon0->SurfaceArea() << " has to be 41.6585425"
            << std::endl;
  std::cout << "Less Simple Polyhedron(2 cutted piramides) SurfaceArea =" << MyPGon1->SurfaceArea()
            << " has to be 65.941..." << std::endl;
  // Asserts
  Vec_t p1, p2, p3, p4, p5, p6, p7, p8, dirx, diry, dirz;
  p1 = Vec_t(0, 0, -5);
  p2 = Vec_t(50, 0, 40);
  p3 = Vec_t(5, 1, 20);
  p4 = Vec_t(45, 5, 30);
  p5 = Vec_t(0, 0, 30);
  p6 = Vec_t(41, 0, 10);
  p7 = Vec_t(0, 0, 0);
  p8 = Vec_t(15, 0, 0);

  dirx = Vec_t(1, 0, 0);
  diry = Vec_t(0, 1, 0);
  dirz = Vec_t(0, 0, 1);

  // Check Extent and cached BBox
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  MyPGon->Extent(minExtent, maxExtent);
  MyPGon->GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  std::cout << "polyhedra Extent():  min=" << minExtent << " max=" << maxExtent << std::endl;
  // In VecGeom the extent is minimal
  assert(ApproxEqual(minExtent, Vec_t(0, 0, -30)));
  assert(ApproxEqual(maxExtent, Vec_t(80. / cos(halfdphi), 40. * sqrt(2.) / cos(halfdphi), 40)));
  assert(ApproxEqual(minExtent, minBBox));
  assert(ApproxEqual(maxExtent, maxBBox));

  // Check Inside
  std::cout << " EInside values:  kInside=" << vecgeom::EInside::kInside << ", kSurface=" << vecgeom::EInside::kSurface
            << ", kOutside=" << vecgeom::EInside::kOutside << "\n";
  std::cout << " MyPGon->Inside(" << p1 << ") = " << MyPGon->Inside(p1) << "\n";
  std::cout << " MyPGon->Inside(" << p2 << ") = " << MyPGon->Inside(p2) << "\n";
  std::cout << " MyPGon->Inside(" << p3 << ") = " << MyPGon->Inside(p3) << "\n";
  std::cout << " MyPGon->Inside(" << p4 << ") = " << MyPGon->Inside(p4) << "\n";
  std::cout << " MyPGon->Inside(" << p5 << ") = " << MyPGon->Inside(p5) << "\n";
  std::cout << " MyPGon->Inside(" << p6 << ") = " << MyPGon->Inside(p6) << "\n";
  std::cout << " MyPGon2->Inside(" << p7 << ") = " << MyPGon2->Inside(p7) << "\n";
  std::cout << " MyPGon2->Inside(" << p8 << ") = " << MyPGon2->Inside(p8) << "\n";

  assert(MyPGon->Inside(p1) == vecgeom::EInside::kSurface);
  assert(MyPGon->Inside(p2) == vecgeom::EInside::kSurface);
  assert(MyPGon->Inside(p3) == vecgeom::EInside::kInside);
  assert(MyPGon->Inside(p4) == vecgeom::EInside::kInside);
  assert(MyPGon->Inside(p5) == vecgeom::EInside::kOutside);
  assert(MyPGon->Inside(p6) == vecgeom::EInside::kOutside);
  assert(MyPGon2->Inside(p7) == vecgeom::EInside::kInside);
  assert(MyPGon2->Inside(p8) == vecgeom::EInside::kSurface);

  // Check that Inside and Contains agree for points around phi tolerance.
  {
    // Note: the point below is at 2 * kTolerance / sqrt(2) distance from boundary, so inside
    Vec_t pPhiInside(20 + 2 * tolerance, 20, 5);
    std::cout << " MyPGon->Contains(" << pPhiInside << ") = " << MyPGon->Contains(pPhiInside) << "\n";
    std::cout << " MyPGon->Inside(" << pPhiInside << ") = " << MyPGon->Inside(pPhiInside) << "\n";

    assert(MyPGon->Contains(pPhiInside));
    assert(MyPGon->Inside(pPhiInside) == vecgeom::EInside::kInside);
  }

  {
    Vec_t pPhiSurface(20, 20, 5);
    std::cout << " MyPGon->Contains(" << pPhiSurface << ") = " << MyPGon->Contains(pPhiSurface) << "\n";
    std::cout << " MyPGon->Inside(" << pPhiSurface << ") = " << MyPGon->Inside(pPhiSurface) << "\n";

    assert(MyPGon->Inside(pPhiSurface) == vecgeom::EInside::kSurface);
  }

  {
    // Note: the point below is at 2 * kTolerance / sqrt(2) distance from boundary, so outside
    Vec_t pPhiOutside(20 - 2 * tolerance, 20, 5);
    std::cout << " MyPGon->Contains(" << pPhiOutside << ") = " << MyPGon->Contains(pPhiOutside) << "\n";
    std::cout << " MyPGon->Inside(" << pPhiOutside << ") = " << MyPGon->Inside(pPhiOutside) << "\n";

    assert(!MyPGon->Contains(pPhiOutside));
    assert(MyPGon->Inside(pPhiOutside) == vecgeom::EInside::kOutside);
  }

  // Check DistanceToIn
  assert(std::fabs((MyPGon->DistanceToIn(p1, dirx))) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p1, -diry))) < tolerance);
  // Point on top endcap moving horizontally: either enter at 0 or not enter at all
  assert(std::fabs((MyPGon->DistanceToIn(p2, diry))) < tolerance ||
         std::fabs((MyPGon->DistanceToIn(p2, diry))) > 1.E10);
  assert(std::fabs((MyPGon->DistanceToIn(p5, dirx) - 40.12368793931)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p6, -dirx) - 0.87631206069)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p6, dirz) - 0.218402670765)) < tolerance);

  // Check DistanceToOut
  Vec_t normal;
  assert(std::fabs((MyPGon->DistanceToOut(p1, -dirx))) < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p3, -diry) - 1.)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p3, dirz) - 1.27382374146)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p4, dirz) - 10.)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p4, dirx) - 34.8538673445)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p4, diry) - 40.)) < tolerance);
  assert(MyPGon2->DistanceToOut(p7, dirx) > 0);
  assert(MyPGon2->DistanceToOut(p7, diry) > 0);
  assert(MyPGon2->DistanceToOut(p7, dirz) > 0);
  std::cout << "MyPGon2->DistanceToOut(p7, dirx) = " << MyPGon2->DistanceToOut(p7, dirx) << std::endl;
  std::cout << "MyPGon2->DistanceToOut(p7, diry) = " << MyPGon2->DistanceToOut(p7, diry) << std::endl;
  std::cout << "MyPGon2->DistanceToOut(p7, dirz) = " << MyPGon2->DistanceToOut(p7, dirz) << std::endl;

#ifdef SCAN_SOLID
  std::cout << "\n=======     Polyhedra SCAN test      ========";
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
    in = MyPGon->Inside(pt);

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
  Vec_t dir(1. / std::sqrt(2.), 1. / std::sqrt(2.), 0);
  Precision d;
  int z;

  std::cout << "\nPdep is (0, 0, z)";
  std::cout << "\nDir is (1, 1, 0)\n";

  for (z = -30; z <= 50; z += 5) {
    // start.setZ(z);
    start.Set(0, 0, z);

    in = MyPGon->Inside(start);
    std::cout << "x=" << start.x() << "  y=" << start.y() << "  z=" << start.z();

    if (in == vecgeom::EInside::kInside) {
      std::cout << " is inside";

      d = MyPGon->DistanceToOut(start, dir);
      std::cout << "  distance to out=" << d;
      d = MyPGon->SafetyToOut(start);
      std::cout << "  closest distance to out=" << d << std::endl;
    } else if (in == vecgeom::EInside::kOutside) {
      std::cout << " is outside";

      d = MyPGon->DistanceToIn(start, dir);
      std::cout << "  distance to in=" << d;
      d = MyPGon->SafetyToIn(start);
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
    d2 = MyPGon->DistanceToIn(start2, dir2);
    std::cout << "  distance to in=" << d2;
    d2 = MyPGon->SafetyToIn(start2);
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
    d3 = MyPGon->DistanceToIn(start3, dir3);
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
    // G4Precision phi=pi/180.*rad;
    //  G4Precision phi=0.0000000001*pi/180.*rad;
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
      in = MyPGon->Inside(start4);
      if (in == vecgeom::EInside::kInside) {
        std::cout << " is inside";
        d4 = MyPGon->DistanceToOut(start4, dir4);
        std::cout << "  distance to out=" << d4;
        d4 = MyPGon->SafetyToOut(start4);
        std::cout << " closest distance to out=" << d4 << std::endl;
      } else if (in == vecgeom::EInside::kOutside) {
        std::cout << " is outside";
        d4 = MyPGon->DistanceToIn(start4, dir4);
        std::cout << "  distance to in=" << d4;
        d4 = MyPGon->SafetyToIn(start4);
        std::cout << " closest distance to in=" << d4 << std::endl;
      } else {
        std::cout << " is on the surface";
        d4 = MyPGon->DistanceToIn(start4, dir4);
        std::cout << "  distance to in=" << d4;
        d4 = MyPGon->SafetyToIn(start4);
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
      in = MyPGon->Inside(start5);
      if (in == vecgeom::EInside::kInside) {
        std::cout << " is inside";
        d5 = MyPGon->DistanceToOut(start5, dir5);
        std::cout << "  distance to out=" << d5;
        d5 = MyPGon->SafetyToOut(start5);
        std::cout << " closest distance to out=" << d5 << std::endl;
      } else if (in == vecgeom::EInside::kOutside) {
        std::cout << " is outside";
        d5 = MyPGon->DistanceToIn(start5, dir5);
        std::cout << "  distance to in=" << d5;
        d5 = MyPGon->SafetyToIn(start5);
        std::cout << " closest distance to in=" << d5 << std::endl;
      } else {
        std::cout << " is on the surface";
        d5 = MyPGon->DistanceToIn(start5, dir5);
        std::cout << "  distance to in=" << d5;
        d5 = MyPGon->SafetyToIn(start5);
        std::cout << " closest distance to in=" << d5 << std::endl;
      }
    }
  }
#endif

  return true;
}

int main(int argc, char *argv[])
{
  assert(TestPolyhedra<vecgeom::SimplePolyhedron>());
  std::cout << "VecGeom Polyhedron passed\n";

  return 0;
}
