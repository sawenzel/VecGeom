// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for the Elliptical Tube
/// @file test/unit_teststest/TestEllipticalTube.cpp
/// @author Evgueni Tcherniaev

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/EllipticalTube.h"
#include "ApproxEqual.h"

bool testvecgeom = false;

using vecgeom::kInfLength;
using vecgeom::kTolerance;

template <class EllipticalTube_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestEllipticalTube()
{
  // Check surfce area and volume
  //
  std::cout << "=== Check Set/Get, Print(), SurfaceArea(), Capacity(), Extent()" << std::endl;

  EllipticalTube_t tube("Test_Elliptical_Tube", 1., 2., 3.);
  VECGEOM_ASSERT(tube.GetDx() == 1.);
  VECGEOM_ASSERT(tube.GetDy() == 2.);
  VECGEOM_ASSERT(tube.GetDz() == 3.);

  tube.SetParameters(4., 5., 6.);
  VECGEOM_ASSERT(tube.GetDx() == 4.);
  VECGEOM_ASSERT(tube.GetDy() == 5.);
  VECGEOM_ASSERT(tube.GetDz() == 6.);

  tube.SetDx(7.);
  tube.SetDy(8.);
  tube.SetDz(9.);
  VECGEOM_ASSERT(tube.GetDx() == 7.);
  VECGEOM_ASSERT(tube.GetDy() == 8.);
  VECGEOM_ASSERT(tube.GetDz() == 9.);

  double a, b, z;
  tube.SetParameters(a = 4., b = 3., z = 5.);
  std::cout << "EllipticalTube(" << a << ", " << b << ", " << z << ")" << std::endl;
  VECGEOM_ASSERT(a >= b);

  double area = tube.SurfaceArea();
  std::cout << "Area : " << area << std::endl;
  double sbase = vecgeom::kPi * a * b;
  double sside = 2. * z * vecgeom::EllipticUtilities::EllipsePerimeter(a, b);
  VECGEOM_ASSERT(area == 2. * sbase + sside);

  double vol = tube.Capacity();
  std::cout << "Volume : " << vol << std::endl;
  VECGEOM_ASSERT(vol == 2. * sbase * z);

  Vec_t bmin, bmax;
  tube.Extent(bmin, bmax);
  std::cout << "Extent : " << bmin << ", " << bmax << std::endl;
  VECGEOM_ASSERT(bmax == Vec_t(a, b, z));
  VECGEOM_ASSERT(bmin == -bmax);

  // Check Inside()
  //
  std::cout << "=== Check Inside()" << std::endl;
  int izmax    = 20;                        // number of steps along Z
  int iphimax  = 36;                        // number of steps along Phi
  int iscmax   = 1000;                      // number of scale factors
  double dz    = 2. * z / izmax;            // step along Z
  double dphi  = vecgeom::kTwoPi / iphimax; // step along Phi
  double dsc   = 1. / iscmax;               // scale factor increment
  double delta = 0.999 * 0.5 * kTolerance;  // shift within surface
  double error = kTolerance / 1000.;        // calculation error tolerance

  // Check inside points
  for (int iz = 0; iz <= izmax; ++iz) {
    double curz = iz * dz - z;
    for (int iphi = 0; iphi < iphimax; ++iphi) {
      double phi  = iphi * dphi;
      double curx = a * std::cos(phi);
      double cury = b * std::sin(phi);
      for (double isc = 0; isc < iscmax; ++isc) {
        double scale = isc * dsc;
        if (tube.Inside(Vec_t(curx, cury, curz) * scale) != vecgeom::kInside) {
          std::cout << "iphi, iz, scale = " << iphi << ", " << iz << ", " << scale << std::endl;
          std::cout << "Point = " << Vec_t(curx, cury, curz) * scale << std::endl;
          VECGEOM_ASSERT(tube.Inside(Vec_t(curx, cury, curz) * scale) == vecgeom::kInside);
        }
      }
    }
  }

  // Check points on surface
  for (int iphi = 0; iphi < iphimax; ++iphi) {
    double phi  = iphi * dphi;
    double curx = (a + delta) * std::cos(phi);
    double cury = (b + delta) * std::sin(phi);
    for (double isc = 0; isc <= iscmax; ++isc) {
      double scale = isc * dsc;

      // base at -Z
      if (tube.Inside(Vec_t(curx * scale, cury * scale, -z)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, -z) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, -z)) == vecgeom::kSurface);
      }
      if (tube.Inside(Vec_t(curx * scale, cury * scale, -z - delta)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, -z - delta) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, -z - delta)) == vecgeom::kSurface);
      }
      if (tube.Inside(Vec_t(curx * scale, cury * scale, -z + delta)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, -z + delta) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, -z + delta)) == vecgeom::kSurface);
      }

      // base at +Z
      if (tube.Inside(Vec_t(curx * scale, cury * scale, z)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, z) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, z)) == vecgeom::kSurface);
      }
      if (tube.Inside(Vec_t(curx * scale, cury * scale, z - delta)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, z - delta) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, z - delta)) == vecgeom::kSurface);
      }
      if (tube.Inside(Vec_t(curx * scale, cury * scale, z + delta)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, z + delta) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, z + delta)) == vecgeom::kSurface);
      }
    }
  }

  // lateral surface
  for (int iphi = 0; iphi < iphimax; ++iphi) {
    double phi  = iphi * dphi;
    double curx = a * std::cos(phi);
    double cury = b * std::sin(phi);
    for (int iz = 0; iz <= izmax; ++iz) {
      double curz = iz * dz - z;
      if (tube.Inside(Vec_t(curx, cury, curz)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx, cury, curz) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curz, cury, curz)) == vecgeom::kSurface);
      }
    }
  }
  for (int iphi = 0; iphi < iphimax; ++iphi) {
    double phi  = iphi * dphi;
    double curx = (a + delta) * std::cos(phi);
    double cury = (b + delta) * std::sin(phi);
    for (int iz = 0; iz <= izmax; ++iz) {
      double curz = iz * dz - z;
      if (tube.Inside(Vec_t(curx, cury, curz)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx, cury, curz) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curz, cury, curz)) == vecgeom::kSurface);
      }
    }
  }
  for (int iphi = 0; iphi < iphimax; ++iphi) {
    double phi  = iphi * dphi;
    double curx = (a - delta) * std::cos(phi);
    double cury = (b - delta) * std::sin(phi);
    for (int iz = 0; iz <= izmax; ++iz) {
      double curz = iz * dz - z;
      if (tube.Inside(Vec_t(curx, cury, curz)) != vecgeom::kSurface) {
        std::cout << "Point = " << Vec_t(curx, cury, curz) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curz, cury, curz)) == vecgeom::kSurface);
      }
    }
  }

  // Check outside points
  for (int iphi = 0; iphi < iphimax; ++iphi) {
    double phi  = iphi * dphi;
    double curx = (a + kTolerance) * std::cos(phi);
    double cury = (b + kTolerance) * std::sin(phi);
    for (double isc = 0; isc <= iscmax; ++isc) {
      double scale = isc * dsc;
      // near base at -Z
      if (tube.Inside(Vec_t(curx * scale, cury * scale, -z - kTolerance)) != vecgeom::kOutside) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, -z - kTolerance) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, -z - kTolerance)) == vecgeom::kOutside);
      }
      // near base at +Z
      if (tube.Inside(Vec_t(curx * scale, cury * scale, z + kTolerance)) != vecgeom::kOutside) {
        std::cout << "Point = " << Vec_t(curx * scale, cury * scale, z + kTolerance) << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx * scale, cury * scale, z + kTolerance)) == vecgeom::kOutside);
      }
    }
  }
  for (int iz = 0; iz <= izmax; ++iz) {
    double curz = iz * dz - z;
    for (int iphi = 0; iphi < iphimax; ++iphi) {
      double phi  = iphi * dphi;
      double curx = a * std::cos(phi);
      double cury = b * std::sin(phi);
      // around lateral surface
      if (tube.Inside(Vec_t(curx, cury, curz) * 1.001) != vecgeom::kOutside) {
        std::cout << "Point = " << Vec_t(curx, cury, curz) * 1.001 << std::endl;
        VECGEOM_ASSERT(tube.Inside(Vec_t(curx, cury, curz) * 1.001) == vecgeom::kOutside);
      }
    }
  }

  // Check Normal()
  //
  std::cout << "=== Check Normal()" << std::endl;
  Vec_t normal(0.);
  bool valid;

  // points on elliptical section
  double scaleX = tube.GetDx();
  double scaleY = tube.GetDy();
  for (int i = 0; i < 360; ++i) {
    double phi = i * vecgeom::kPi / 180;
    double nx  = std::cos(phi);
    double ny  = std::sin(phi);
    double px  = nx * scaleX;
    double py  = ny * scaleY;
    valid      = tube.Normal(Vec_t(px, py, 0), normal);
    VECGEOM_ASSERT(normal == Vec_t(nx * scaleY, ny * scaleX, 0).Unit());
  }

  // points near axes
  for (int i = 0; i < 21; ++i) {
    double curx = tube.GetDx() + 0.1 * kTolerance * (i - 10);
    valid       = tube.Normal(Vec_t(curx, 0, 0), normal);
    VECGEOM_ASSERT(normal == Vec_t(1, 0, 0));
    valid = tube.Normal(Vec_t(-curx, 0, 0), normal);
    VECGEOM_ASSERT(normal == Vec_t(-1, 0, 0));

    double cury = tube.GetDy() + 0.1 * kTolerance * (i - 10);
    valid       = tube.Normal(Vec_t(0, cury, 0), normal);
    VECGEOM_ASSERT(normal == Vec_t(0, 1, 0));
    valid = tube.Normal(Vec_t(0, -cury, 0), normal);
    VECGEOM_ASSERT(normal == Vec_t(0, -1, 0));

    double curz = tube.GetDz() + 0.1 * kTolerance * (i - 10);
    valid       = tube.Normal(Vec_t(0, 0, curz), normal);
    VECGEOM_ASSERT(normal == Vec_t(0, 0, 1));
    valid = tube.Normal(Vec_t(0, 0, -curz), normal);
    VECGEOM_ASSERT(normal == Vec_t(0, 0, -1));
  }

  // point on edge
  valid = tube.Normal(Vec_t(tube.GetDx(), 0, tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(1, 0, 1).Unit());
  valid = tube.Normal(Vec_t(0, tube.GetDy(), tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(0, 1, 1).Unit());
  valid = tube.Normal(Vec_t(-tube.GetDx(), 0, tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(-1, 0, 1).Unit());
  valid = tube.Normal(Vec_t(0, -tube.GetDy(), tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(0, -1, 1).Unit());

  valid = tube.Normal(Vec_t(tube.GetDx(), 0, -tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(1, 0, -1).Unit());
  valid = tube.Normal(Vec_t(0, tube.GetDy(), -tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(0, 1, -1).Unit());
  valid = tube.Normal(Vec_t(-tube.GetDx(), 0, -tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(-1, 0, -1).Unit());
  valid = tube.Normal(Vec_t(0, -tube.GetDy(), -tube.GetDz()), normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(normal == Vec_t(0, -1, -1).Unit());

  // special case of point on Z-axis
  valid = tube.Normal(Vec_t(0, 0, 0), normal);
  VECGEOM_ASSERT(!valid);
  VECGEOM_ASSERT(normal == Vec_t(0, 0, 1));
  valid = tube.Normal(Vec_t(0, 0, -kTolerance), normal);
  VECGEOM_ASSERT(!valid);
  VECGEOM_ASSERT(normal == Vec_t(0, 0, -1));

  // Check SafetyToIn()
  //
  std::cout << "=== Check SafetyToIn()" << std::endl;
  VECGEOM_ASSERT(a >= b);
  for (int iz = 1; iz < izmax; ++iz) {
    double curz = iz * dz - z;
    for (int iphi = 0; iphi < iphimax; ++iphi) {
      double phi  = iphi * dphi;
      double curx = 2. * a * std::cos(phi);
      double cury = 2. * b * std::sin(phi);
      if (tube.SafetyToIn(Vec_t(curx, cury, curz)) < b - error) {
        std::cout << "SafetyToIn" << Vec_t(curx, cury, curz) << " = " << std::setprecision(16)
                  << tube.SafetyToIn(Vec_t(curx, cury, curz)) << std::endl;
        VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(curx, cury, curz)) >= b - error);
      }
      curx = (a + delta) * std::cos(phi);
      cury = (b - delta) * std::sin(phi);
      if (tube.SafetyToIn(Vec_t(curx, cury, curz)) != 0) {
        std::cout << "SafetyToIn" << Vec_t(curx, cury, curz) << " = " << std::setprecision(16)
                  << tube.SafetyToIn(Vec_t(curx, cury, curz)) << std::endl;
        VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(curx, cury, curz)) == 0);
      }
      curx = 0.999 * a * std::cos(phi);
      cury = 0.999 * b * std::sin(phi);
      if (tube.SafetyToIn(Vec_t(curx, cury, curz)) >= 0) {
        std::cout << "SafetyToIn" << Vec_t(curx, cury, curz) << " = " << std::setprecision(16)
                  << tube.SafetyToIn(Vec_t(curx, cury, curz)) << std::endl;
        VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(curx, cury, curz)) < 0);
      }
    }
  }

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, 0)) < 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 2 * b, 0)) == b);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(2 * a, 0, 0)) >= b);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(a, 0, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(a - delta, 0, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(a + delta, 0, 0)) == 0);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, b, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, b - delta, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, b + delta, 0)) == 0);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(a / 2, b / 2, 0)) < 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(a / 2, 0, 0)) < 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, b / 2, 0)) < 0);

  // points around the bases
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, -z - 1)) == 1);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, z + 2)) == 2);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, -z)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, -z - delta)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, -z + delta)) == 0);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, z)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, z - delta)) == 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, z + delta)) == 0);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, -z + 1)) < 0);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, 0, z - 2)) < 0);

  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(-3 * a, 0, z + 10)) == 10);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, -3 * b, z + 10)) == 10);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(-3 * a, 0, -z - 1)) >= 2 * b);
  VECGEOM_ASSERT(tube.SafetyToIn(Vec_t(0, -3 * b, -z - 2)) == 2 * b);

  // Check SafetyToOut()
  //
  std::cout << "=== Check SafetyToOut()" << std::endl;
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, 0)) == b);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, b / 2, 0)) == b / 2);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(a / 2, 0, 0)) >= b / 2);

  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(a, 0, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(a - delta, 0, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(a + delta, 0, 0)) == 0);

  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, b, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, b - delta, 0)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, b + delta, 0)) == 0);

  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(2 * a, 2 * b, 0)) < 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(-a, -b, z / 2)) < 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(2 * a, 0, 0)) < 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 2 * b, 0)) < 0);

  // points around the bases
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, -z + 1)) == 1);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, z - 2)) == 2);

  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, -z)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, -z - delta)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, -z + delta)) == 0);

  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, z)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, z - delta)) == 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, z + delta)) == 0);

  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, -z - 1)) < 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, 0, z + 2)) < 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(a, 0, -z - 1)) < 0);
  VECGEOM_ASSERT(tube.SafetyToOut(Vec_t(0, b, z + 2)) < 0);

  // Check DistanceToIn()
  //
  std::cout << "=== Check DistanceToIn()" << std::endl;

  tube.SetParameters(a = 4., b = 3., z = 5.);
  double Rsph = std::sqrt(a * a + b * b + z * z) + 0.1; // surrounding sphere
  double dist, del = kTolerance / 3.;
  Vec_t pnt, dir;

  // set coordinates for points in grid
  static const int np = 11;
  double xxx[np]      = {-a - 1, -a - del, -a, -a + del, -1, 0, 1, a - del, a, a + del, a + 1};
  double yyy[np]      = {-b - 1, -b - del, -b, -b + del, -1, 0, 1, b - del, b, b + del, b + 1};
  double zzz[np]      = {-z - 1, -z - del, -z, -z + del, -1, 0, 1, z - del, z, z + del, z + 1};

  // check directions parallel to Z axis
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        pnt.Set(xxx[ix], yyy[iy], zzz[iz]);
        Vec_t rho(pnt.x(), pnt.y(), 0);

        dist = tube.DistanceToIn(pnt, dir = Vec_t(0, 0, 1));
        if (tube.Inside(pnt) == vecgeom::kInside) VECGEOM_ASSERT(dist < 0);
        if (tube.Inside(rho) != vecgeom::kInside) {
          VECGEOM_ASSERT(dist == kInfLength);
        } else {
          if (tube.Inside(pnt) != vecgeom::kInside && pnt.z() > 0) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            VECGEOM_ASSERT(dist == -pnt.z() - z);
          }
        }

        dist = tube.DistanceToIn(pnt, dir = Vec_t(0, 0, -1));
        if (tube.Inside(pnt) == vecgeom::kInside) VECGEOM_ASSERT(dist < 0);
        if (tube.Inside(rho) != vecgeom::kInside) {
          VECGEOM_ASSERT(dist == kInfLength);
        } else {
          if (tube.Inside(pnt) != vecgeom::kInside && pnt.z() < 0) {
            VECGEOM_ASSERT(dist == kInfLength);
          } else {
            VECGEOM_ASSERT(dist == pnt.z() - z);
          }
        }
      }
    }
  }

  // check directions parallel to X axis
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        pnt.Set(xxx[ix], yyy[iy], zzz[iz]);

        Vec_t height(0, 0, pnt.z());
        double tmp  = (1 - (pnt.y() / b)) * (1 + (pnt.y() / b));
        double intx = (tmp < 0) ? 0 : std::sqrt(tmp) * a;

        dist = tube.DistanceToIn(pnt, dir = Vec_t(1, 0, 0));
        if (tube.Inside(pnt) == vecgeom::kInside) VECGEOM_ASSERT(dist < 0);
        if (tube.Inside(pnt) == vecgeom::kInside) continue;
        if (tube.Inside(height) != vecgeom::kInside) VECGEOM_ASSERT(dist == vecgeom::kInfLength);
        if (tube.Inside(height) != vecgeom::kInside) continue;
        if (tmp <= 0) VECGEOM_ASSERT(dist == kInfLength);
        if (tmp <= 0) continue;

        if (pnt.x() > 0) VECGEOM_ASSERT(dist == kInfLength);
        if (pnt.x() < 0) VECGEOM_ASSERT(ApproxEqual(dist, std::abs(pnt.x()) - intx));

        dist = tube.DistanceToIn(pnt, dir = Vec_t(-1, 0, 0));
        if (pnt.x() < 0) VECGEOM_ASSERT(dist == kInfLength);
        if (pnt.x() > 0) VECGEOM_ASSERT(ApproxEqual(dist, std::abs(pnt.x()) - intx));
      }
    }
  }

  // Check inside points ("wrong side")
  for (int ith = 5; ith < 180; ith += 30) {
    for (int iph = 5; iph < 360; iph += 30) {
      double theta = ith * vecgeom::kPi / 180;
      double phi   = iph * vecgeom::kPi / 180;
      double vx    = std::sin(theta) * std::cos(phi);
      double vy    = std::sin(theta) * std::sin(phi);
      double vz    = std::cos(theta);
      dir.Set(vx, vy, vz);
      dist = tube.DistanceToIn(Vec_t(0, 0, 0), dir);
      VECGEOM_ASSERT(dist < 0);
    }
  }

  // Check outside points
  for (int ith = 5; ith < 180; ith += 30) {
    for (int iph = 5; iph < 360; iph += 30) {
      double theta = ith * vecgeom::kPi / 180;
      double phi   = iph * vecgeom::kPi / 180;
      double vx    = std::sin(theta) * std::cos(phi);
      double vy    = std::sin(theta) * std::sin(phi);
      double vz    = std::cos(theta);
      dir.Set(vx, vy, vz); // direction
      pnt = Rsph * dir;

      Vec_t axis(1, 0, 0);
      if (std::abs(vy) < std::abs(vx) && std::abs(vy) < std::abs(vz)) axis.Set(0, 1, 0);
      if (std::abs(vz) < std::abs(vx) && std::abs(vz) < std::abs(vy)) axis.Set(0, 0, 1);
      Vec_t ort = (axis.Cross(dir)).Unit(); // orthogonal direction

      dist = tube.DistanceToIn(pnt, dir);
      VECGEOM_ASSERT(dist == kInfLength);

      dist = tube.DistanceToIn(pnt, ort);
      VECGEOM_ASSERT(dist == kInfLength);

      dist = tube.DistanceToIn(pnt, -ort);
      VECGEOM_ASSERT(dist == kInfLength);

      dist = tube.DistanceToIn(pnt, -dir);
      VECGEOM_ASSERT(dist > 0 && dist < Rsph);
      VECGEOM_ASSERT(tube.Inside(pnt - dist * dir) == vecgeom::kSurface);
    }
  }

  // Check "very far" outside points
  double offset = 1.e+6;
  for (int ith = 0; ith <= 180; ith += 10) {
    for (int iph = 0; iph < 360; iph += 10) {
      double theta = ith * vecgeom::kPi / 180;
      double phi   = iph * vecgeom::kPi / 180;
      double vx    = std::sin(theta) * std::cos(phi);
      double vy    = std::sin(theta) * std::sin(phi);
      double vz    = std::cos(theta);
      dir.Set(vx, vy, vz); // direction
      pnt = (offset + Rsph) * dir;

      Vec_t axis(1, 0, 0);
      if (std::abs(vy) < std::abs(vx) && std::abs(vy) < std::abs(vz)) axis.Set(0, 1, 0);
      if (std::abs(vz) < std::abs(vx) && std::abs(vz) < std::abs(vy)) axis.Set(0, 0, 1);
      Vec_t ort = (axis.Cross(dir)).Unit(); // orthogonal direction

      dist = tube.DistanceToIn(pnt, dir);
      VECGEOM_ASSERT(dist == kInfLength);

      dist = tube.DistanceToIn(pnt, ort);
      VECGEOM_ASSERT(dist == kInfLength);

      dist = tube.DistanceToIn(pnt, -ort);
      VECGEOM_ASSERT(dist == kInfLength);

      dist = tube.DistanceToIn(pnt, -dir);
      VECGEOM_ASSERT(tube.Inside(pnt - dist * dir) == vecgeom::kSurface);
    }
  }

  // Check DistanceToOut()
  //
  std::cout << "=== Check DistanceToOut()" << std::endl;

  // check directions parallel to Z axis
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        pnt.Set(xxx[ix], yyy[iy], zzz[iz]);

        dist = tube.DistanceToOut(pnt, dir = Vec_t(0, 0, 1));
        if (tube.Inside(pnt) == vecgeom::kOutside) VECGEOM_ASSERT(dist == -1);
        if (tube.Inside(pnt) != vecgeom::kOutside) VECGEOM_ASSERT(dist == (z - pnt.z()));

        dist = tube.DistanceToOut(pnt, dir = Vec_t(0, 0, -1));
        if (tube.Inside(pnt) == vecgeom::kOutside) VECGEOM_ASSERT(dist == -1);
        if (tube.Inside(pnt) != vecgeom::kOutside) VECGEOM_ASSERT(dist == (z + pnt.z()));
      }
    }
  }

  // check directions parallel to X axis
  for (int ix = 0; ix < np; ++ix) {
    for (int iy = 0; iy < np; ++iy) {
      for (int iz = 0; iz < np; ++iz) {
        pnt.Set(xxx[ix], yyy[iy], zzz[iz]);

        double tmp  = (1 - (pnt.y() / b)) * (1 + (pnt.y() / b));
        double intx = (tmp < 0) ? 0 : std::sqrt(tmp) * a;

        dist = tube.DistanceToOut(pnt, dir = Vec_t(1, 0, 0));
        if (tube.Inside(pnt) == vecgeom::kOutside) VECGEOM_ASSERT(dist < 0);
        if (tube.Inside(pnt) == vecgeom::kOutside) continue;
        if (tmp <= 0) VECGEOM_ASSERT(dist == 0);
        if (tmp > 0) VECGEOM_ASSERT(ApproxEqual(dist, intx - pnt.x()));

        dist = tube.DistanceToOut(pnt, dir = Vec_t(-1, 0, 0));
        if (tmp <= 0) VECGEOM_ASSERT(dist == 0);
        if (tmp > 0) VECGEOM_ASSERT(ApproxEqual(dist, intx + pnt.x()));
      }
    }
  }

  // Check inside points
  for (int ith = 5; ith < 180; ith += 30) {
    for (int iph = 5; iph < 360; iph += 30) {
      double theta = ith * vecgeom::kPi / 180;
      double phi   = iph * vecgeom::kPi / 180;
      double vx    = std::sin(theta) * std::cos(phi);
      double vy    = std::sin(theta) * std::sin(phi);
      double vz    = std::cos(theta);
      dir.Set(vx, vy, vz);
      dist = tube.DistanceToOut(Vec_t(0, 0, 0), dir);
      VECGEOM_ASSERT(dist > 0 && dist < Rsph);
      VECGEOM_ASSERT(tube.Inside(dist * dir) == vecgeom::kSurface);
    }
  }

  // Check outside points ("wrong side")
  for (int ith = 5; ith < 180; ith += 30) {
    for (int iph = 5; iph < 360; iph += 30) {
      double theta = ith * vecgeom::kPi / 180;
      double phi   = iph * vecgeom::kPi / 180;
      double vx    = std::sin(theta) * std::cos(phi);
      double vy    = std::sin(theta) * std::sin(phi);
      double vz    = std::cos(theta);
      dir.Set(vx, vy, vz);

      dist = tube.DistanceToOut(Rsph * dir, dir);
      VECGEOM_ASSERT(dist == -1.); // convention: VECGEOM_ASSERT(dist < 0);

      dist = tube.DistanceToOut(Rsph * dir, -dir);
      VECGEOM_ASSERT(dist == -1.); // convention: VECGEOM_ASSERT(dist < 0);
    }
  }

  // Check SamplePointOnSurface()
  //
  std::cout << "=== Check SamplePointOnSurface()" << std::endl;
  tube.SetParameters(a = 4., b = 3., z = 5.); // set cylinder

  int nzneg = 0, nzpos = 0, nside = 0, nfactor = 10000, ntot = 4 * area * nfactor;
  for (int i = 0; i < ntot; i++) {
    Vec_t rndPoint = tube.GetUnplacedVolume()->SamplePointOnSurface();
    VECGEOM_ASSERT(tube.Inside(rndPoint) == vecgeom::EInside::kSurface);
    if (rndPoint.x() < 0 || rndPoint.y() < 0) continue;
    if (rndPoint.z() == -z)
      ++nzneg;
    else if (rndPoint.z() == z)
      ++nzpos;
    else
      ++nside;
  }
  std::cout << "szneg,sside,szpos = " << sbase << ", \t" << sside << ", \t" << sbase << std::endl;
  std::cout << "nzneg,nside,nzpos = " << nzneg << ", \t" << nside << ", \t" << nzpos << std::endl;
  VECGEOM_ASSERT(std::abs(nzneg - sbase * nfactor) < 2. * std::sqrt(ntot));
  VECGEOM_ASSERT(std::abs(nside - sside * nfactor) < 2. * std::sqrt(ntot));
  VECGEOM_ASSERT(std::abs(nzpos - sbase * nfactor) < 2. * std::sqrt(ntot));

  return true;
}

int main(int argc, char *argv[])
{
  VECGEOM_ASSERT(TestEllipticalTube<vecgeom::SimpleEllipticalTube>());
  std::cout << "VecGeomEllipticalTube passed\n";

  return 0;
}
