// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for the Elliptical Cone.
/// @file test/unit_teststest/TestEllipticalCone.cpp
/// @author Evgueni Tcherniaev

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/EllipticalCone.h"
#include "ApproxEqual.h"

using namespace vecgeom;

///////////////////////////////////////////////////////////////////////////////
//
// Check DistanceToIn() for set of points, p.y() should be positive (!!!)
//
template <class Vec_t = Vector3D<Precision>>
void CheckDistanceToIn(const SimpleEllipticalCone &cone, double Z)
{
  double b    = cone.GetSemiAxisY();
  double h    = cone.GetZMax();
  double zcut = cone.GetZTopCut();

  // Define directions to test
  Vec_t vx(1, 0, 0);
  Vec_t vy(0, 1, 0);
  Vec_t vz(0, 0, 1);
  Vec_t vleft  = Vec_t(0, b, 1).Unit();
  Vec_t vright = Vec_t(0, -b, 1).Unit();

  double rho  = std::abs(b * (h - Z)); // distance to z axis at z = Z
  double rtop = b * (h - zcut);        // distance to z axis at z = zcut
  double rbot = b * (h + zcut);        // distance to z axis at z = -zcut

  // Define y-positions to test
  // Code below assumes that the positions are positive
  double del    = 0.9 * kHalfTolerance;
  double yy[10] = {rho + 2 * rbot, rho + 1, rho + del, rho, rho - del, 12000., 5.5, 3.5, 0.5, 0.};

  for (int i = 0; i < 10; i++) {
    Vec_t p(0, yy[i], Z);
    double dist;

    // Check vx
    dist = cone.DistanceToIn(p, vx);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      VECGEOM_ASSERT(dist == kInfLength);
    }

    // Check -vx
    dist = cone.DistanceToIn(p, -vx);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      VECGEOM_ASSERT(dist == kInfLength);
    }

    // Check vy
    dist = cone.DistanceToIn(p, vy);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      VECGEOM_ASSERT(dist == kInfLength);
    }

    // Check -vy
    dist = cone.DistanceToIn(p, -vy);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (std::abs(p.z()) > zcut - kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        VECGEOM_ASSERT(std::abs(dist - (p.y() - rho)) < 0.01 * kTolerance);
      }
    }

    // Check vz
    dist = cone.DistanceToIn(p, vz);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      double abspy = std::abs(p.y());
      if (p.z() > zcut - kHalfTolerance || abspy > rho - kHalfTolerance || abspy > rbot - kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        VECGEOM_ASSERT(std::abs(dist - (-zcut - p.z())) < 0.01 * kTolerance);
      }
    }

    // Check -vz
    dist = cone.DistanceToIn(p, -vz);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() < -zcut + kHalfTolerance || std::abs(p.y()) > rbot - kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        if (p.y() >= -rtop && p.y() <= rtop) {
          VECGEOM_ASSERT(std::abs(dist - (p.z() - zcut)) < 0.01 * kTolerance);
        } else {
          VECGEOM_ASSERT(std::abs(dist - (p.z() - (h - std::abs(p.y()) / b))) < 0.01 * kTolerance);
        }
      }
    }

    // Check vright (\)
    dist = cone.DistanceToIn(p, vright);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() > zcut - kHalfTolerance || p.y() > rho - kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        double dz = (-zcut - p.z());
        double dy = dz * b;
        if (p.y() - dy < -rbot + kHalfTolerance) {
          VECGEOM_ASSERT(dist == kInfLength);
        } else {
          VECGEOM_ASSERT(std::abs(dist - dz * sqrt(1 + b * b)) < 0.01 * kTolerance);
        }
      }
    }

    // Check -vright (\)
    dist = cone.DistanceToIn(p, -vright);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      double B     = vright.z();
      double C     = -vright.y();
      double D1    = -B * rtop - C * zcut;
      double D2    = B * rtop - C * zcut;
      double D3    = B * rbot + C * zcut;
      double dist1 = B * p.y() + C * p.z() + D1;
      double dist2 = B * p.y() + C * p.z() + D2;
      double dist3 = B * p.y() + C * p.z() + D3;
      if (p.z() < -zcut + kHalfTolerance || dist1 > -kHalfTolerance || dist3 < kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        if (dist2 >= 0) {
          double dztop = p.z() - zcut;
          VECGEOM_ASSERT(std::abs(dist - dztop * std::sqrt(1 + b * b)) < 0.01 * kHalfTolerance);
        } else {
          double rr  = -b * (h - p.z());
          double dy  = (rr - p.y()) / 2.;
          double exp = dy * std::sqrt(1 + b * b) / b;
          VECGEOM_ASSERT(std::abs(dist - exp) < kTolerance);
        }
      }
    }

    // Check vleft (/)
    dist = cone.DistanceToIn(p, vleft);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() > zcut - kHalfTolerance || p.y() >= rho - kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        double dz = (-zcut - p.z());
        double dy = dz * b;
        if (p.y() + dy > rbot - kHalfTolerance) {
          VECGEOM_ASSERT(dist == kInfLength);
        } else {
          VECGEOM_ASSERT(std::abs(dist - dz * sqrt(1 + b * b)) < 0.01 * kTolerance);
        }
      }
    }

    // Check -vleft (/)
    dist = cone.DistanceToIn(p, -vleft);
    if (cone.Inside(p) == vecgeom::kInside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      double B     = -vleft.z();
      double C     = vleft.y();
      double Dl    = B * rtop - C * zcut;
      double Dm    = -B * rtop - C * zcut;
      double Dr    = -B * rbot + C * zcut;
      double distl = B * p.y() + C * p.z() + Dl;
      double distm = B * p.y() + C * p.z() + Dm;
      double distr = B * p.y() + C * p.z() + Dr;
      if (p.z() < -zcut + kHalfTolerance || distl > -kHalfTolerance || distr < kHalfTolerance) {
        VECGEOM_ASSERT(dist == kInfLength);
      } else {
        if (distm >= 0) {
          double dztop = p.z() - zcut;
          VECGEOM_ASSERT(std::abs(dist - dztop * std::sqrt(1 + b * b)) < 0.01 * kHalfTolerance);
        } else {
          double rr  = b * (h - p.z());
          double dy  = (p.y() - rr) / 2.;
          double exp = dy * std::sqrt(1 + b * b) / b;
          VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kHalfTolerance);
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//
// Check DistanceToOut() for set of points, p.y() should be positive (!!!)
//
template <class Vec_t = Vector3D<Precision>>
void CheckDistanceToOut(const SimpleEllipticalCone &cone, double Z)
{
  double a    = cone.GetSemiAxisX();
  double b    = cone.GetSemiAxisY();
  double h    = cone.GetZMax();
  double zcut = cone.GetZTopCut();

  // Define directions to test
  Vec_t vx(1, 0, 0);
  Vec_t vy(0, 1, 0);
  Vec_t vz(0, 0, 1);
  Vec_t vleft  = Vec_t(0, b, 1).Unit();
  Vec_t vright = Vec_t(0, -b, 1).Unit();

  double rho  = std::abs(b * (h - Z)); // distance to z axis at z = Z
  double rtop = b * (h - zcut);        // distance to z axis at z = zcut
  double rbot = b * (h + zcut);        // distance to z axis at z = -zcut

  // Define y-positions to test
  // Code below assumes that the positions are positive
  double del   = 0.9 * kHalfTolerance;
  double yy[9] = {rho + 2 * rbot, rho + 1, rho + del, rho, rho - del, 5.5, 3.5, 0.5, 0.};

  for (int i = 0; i < 9; i++) {
    Vec_t p(0, yy[i], Z);
    double dist, exp;

    // Check vx
    dist = cone.DistanceToOut(p, vx);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (std::abs(p.y()) >= rho || p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        exp = a * std::sqrt(((h - p.z()) - p.y() / b) * ((h - p.z()) + p.y() / b));
        VECGEOM_ASSERT(std::abs(dist - exp) < kHalfTolerance);
      }
    }

    // Check -vx
    dist = cone.DistanceToOut(p, -vx);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (std::abs(p.y()) >= rho || p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        exp = a * std::sqrt(((h - Z) - p.y() / b) * ((h - Z) + p.y() / b));
        VECGEOM_ASSERT(std::abs(dist - exp) < kHalfTolerance);
      }
    }

    // Check vy
    dist = cone.DistanceToOut(p, vy);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        exp = rho - p.y();
        VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kHalfTolerance);
      }
    }

    // Check -vy
    dist = cone.DistanceToOut(p, -vy);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        exp = p.y() + rho;
        VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kHalfTolerance);
      }
    }

    // Check vz
    dist = cone.DistanceToOut(p, vz);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        if (std::abs(p.y()) <= rtop) {
          VECGEOM_ASSERT(std::abs(dist - (zcut - p.z())) < 0.01 * kTolerance);
        } else {
          exp = (h - std::abs(p.y()) / b) - p.z();
          VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kTolerance);
        }
      }
    }

    // Check -vz
    dist = cone.DistanceToOut(p, -vz);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      VECGEOM_ASSERT(std::abs(dist - (p.z() + zcut)) < 0.01 * kTolerance);
    }

    // Check vright (\)
    dist = cone.DistanceToOut(p, vright);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        if (p.y() > rho + 0.01 * kHalfTolerance) {
          VECGEOM_ASSERT(dist <= 0);
        } else if (p.y() > rho - 0.01 * kHalfTolerance) {
          double dztop = zcut - p.z();
          exp          = dztop * sqrt(1 + b * b);
          VECGEOM_ASSERT(std::abs(dist) < kHalfTolerance || std::abs(dist - exp) < 0.01 * kHalfTolerance);
        } else if (p.y() > rho - kHalfTolerance) {
          if (zcut < h) {
            double dztop = zcut - p.z();
            exp          = dztop * sqrt(1 + b * b);
            VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kHalfTolerance);
          } else { // zcut == h
            double py    = p.y() / b;
            double pz    = p.z() - h;
            double vy    = vright.y() / b;
            double C     = py * py - pz * pz;
            double B     = py * vy - pz * vright.z();
            double exp1  = -C / (B + B);
            double dztop = zcut - p.z();
            double exp2  = dztop * sqrt(1 + b * b);
            exp          = std::min(exp1, exp2);
            VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kTolerance);
          }
        } else {
          double B   = vright.z(); // line going trough left upper corner at YZ projection
          double C   = -vright.y();
          double D   = B * rtop - C * zcut;
          double del = B * p.y() + C * p.z() + D; // distance to the line
          if (del > 0) {
            double dztop = zcut - p.z();
            exp          = dztop * sqrt(1 + b * b);
            VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kHalfTolerance);
          } else {
            double dy = (p.y() + rho) / 2.;
            exp       = dy * std::sqrt(1 + b * b) / b;
            VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kTolerance);
          }
        }
      }
    }

    // Check -vright (\)
    dist = cone.DistanceToOut(p, -vright);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h && p.y() >= -rho) {
        VECGEOM_ASSERT(dist == 0.);
      } else if (p.y() >= rho) {
        VECGEOM_ASSERT(dist <= 0.);
      } else {
        double dzbot = p.z() + zcut;
        exp          = dzbot * sqrt(1 + b * b);
        VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kTolerance);
      }
    }

    // Check vleft (/)
    dist = cone.DistanceToOut(p, vleft);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h) {
        VECGEOM_ASSERT(dist == 0);
      } else {
        double B   = -vleft.z(); // line going trough right upper corner at YZ projection
        double C   = vleft.y();
        double D   = -B * rtop - C * zcut;
        double del = B * p.y() + C * p.z() + D; // distance to the line
        if (del >= 0.) {
          double dztop = zcut - p.z();
          exp          = dztop * sqrt(1 + b * b);
          VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kHalfTolerance);
        } else {
          double dy = (rho - p.y()) / 2.;
          exp       = dy * std::sqrt(1 + b * b) / b;
          VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kTolerance);
        }
      }
    }

    // Check -vleft (/)
    dist = cone.DistanceToOut(p, -vleft);
    if (cone.Inside(p) == vecgeom::kOutside) {
      VECGEOM_ASSERT(dist < 0);
    } else {
      if (p.z() >= h && p.y() <= rho) {
        VECGEOM_ASSERT(dist == 0.);
      } else {
        double dzbot = p.z() + zcut;
        exp          = dzbot * sqrt(1 + b * b);
        VECGEOM_ASSERT(std::abs(dist - exp) < 0.01 * kTolerance);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//
// Unit test for Elliptical Cone
//
template <class EllipticalCone_t, class Vec_t = Vector3D<Precision>>
bool TestEllipticalCone()
{
  double deg = kPi / 180.;

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check surfce area and volume
  //
  std::cout << "=== Check Set/Get, Print(), SurfaceArea(), Capacity(), Extent()" << std::endl;

  EllipticalCone_t cone("Test_Elliptical_Cone", 1., 2., 3., 4.);
  VECGEOM_ASSERT(cone.GetSemiAxisX() == 1.);
  VECGEOM_ASSERT(cone.GetSemiAxisY() == 2.);
  VECGEOM_ASSERT(cone.GetZMax() == 3.);
  VECGEOM_ASSERT(cone.GetZTopCut() == 3.);

  cone.SetParameters(0.1, 0.2, 10., 8.);
  VECGEOM_ASSERT(cone.GetSemiAxisX() == 0.1);
  VECGEOM_ASSERT(cone.GetSemiAxisY() == 0.2);
  VECGEOM_ASSERT(cone.GetZMax() == 10.);
  VECGEOM_ASSERT(cone.GetZTopCut() == 8.);

  double a, b, h, zcut;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  VECGEOM_ASSERT(a >= b);
  cone.Print();
  std::cout << "EllipticalCone (" << a << ", " << b << ", " << h << ", " << zcut << ")" << std::endl;

  std::cout << std::endl;
  double area = cone.SurfaceArea();
  std::cout << "Area : " << area << std::endl;
  double h1     = h - zcut;
  double h2     = h + zcut;
  double sbase1 = kPi * a * b * h1 * h1;
  double sbase2 = kPi * a * b * h2 * h2;
  double sside1 = EllipticUtilities::EllipticalConeLateralArea(a * h1, b * h1, h1);
  double sside2 = EllipticUtilities::EllipticalConeLateralArea(a * h2, b * h2, h2);
  VECGEOM_ASSERT(ApproxEqual(area, sbase1 + sbase2 + sside2 - sside1));

  double volume = cone.Capacity();
  std::cout << "Volume : " << volume << std::endl;
  VECGEOM_ASSERT(ApproxEqual(volume, sbase2 * h2 / 3. - sbase1 * h1 / 3.));

  Vec_t bmin, bmax;
  cone.Extent(bmin, bmax);
  std::cout << "Extent : " << bmin << ", " << bmax << std::endl;
  VECGEOM_ASSERT(bmax == Vec_t(a * h2, b * h2, zcut));
  VECGEOM_ASSERT(bmin == -bmax);
  std::cout << std::endl;

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check Inside()
  //
  std::cout << "=== Check Inside()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);

  // Check inside points
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      for (double scale = 0.; scale < 0.99; scale += 0.1) {
        double x = scale * a * (h - z) * std::cos(phi);
        double y = scale * b * (h - z) * std::sin(phi);
        VECGEOM_ASSERT(cone.Inside(Vec_t(x, y, z)) == vecgeom::kInside);
      }
    }
  }

  // Check points on lateral surface
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double x = a * (h - z) * std::cos(phi);
      double y = b * (h - z) * std::sin(phi);
      Vec_t p(x, y, z + (0.5 - RNG::Instance().uniform()) * kHalfTolerance);
      VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
    }
  }

  // Check points on bases
  for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
    for (double scale = 0.; scale < 1.01; scale += 0.1) {
      double x1 = scale * a * (h - zcut) * std::cos(phi);
      double y1 = scale * b * (h - zcut) * std::sin(phi);
      double z1 = zcut + (0.5 - RNG::Instance().uniform()) * kHalfTolerance;
      VECGEOM_ASSERT(cone.Inside(Vec_t(x1, y1, z1)) == vecgeom::kSurface);
      double x2 = scale * a * (h + zcut) * std::cos(phi);
      double y2 = scale * b * (h + zcut) * std::sin(phi);
      double z2 = -zcut + (0.5 - RNG::Instance().uniform()) * kHalfTolerance;
      VECGEOM_ASSERT(cone.Inside(Vec_t(x2, y2, z2)) == vecgeom::kSurface);
    }
  }

  // Check outside points
  for (double dz = 0.2, z = -(zcut + dz); z <= (zcut + dz); z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double scale = 1.1;
      double x     = scale * a * (h - z) * std::cos(phi);
      double y     = scale * b * (h - z) * std::sin(phi);
      VECGEOM_ASSERT(cone.Inside(Vec_t(x, y, z)) == vecgeom::kOutside);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check Normal()
  //
  std::cout << "=== Check Normal()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  Vec_t normal(0.);
  bool valid;

  // points on lateral surface
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    VECGEOM_ASSERT(valid = cone.Normal(Vec_t(a * (h - z), 0, z), normal));
    VECGEOM_ASSERT(normal == Vec_t(h, 0, a * h).Unit());
    VECGEOM_ASSERT(valid = cone.Normal(Vec_t(a * (z - h), 0, z), normal));
    VECGEOM_ASSERT(normal == Vec_t(-h, 0, a * h).Unit());
    VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, b * (h - z), z), normal));
    VECGEOM_ASSERT(normal == Vec_t(0, h, b * h).Unit());
    VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, b * (z - h), z), normal));
    VECGEOM_ASSERT(normal == Vec_t(0, -h, b * h).Unit());
  }

  // points on bases
  for (double rho = 0; rho < b * (h - zcut); rho += 0.3) {
    for (double dphi = 30. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double x = rho * std::cos(phi);
      double y = rho * std::sin(phi);
      VECGEOM_ASSERT(valid = cone.Normal(Vec_t(x, y, zcut), normal));
      VECGEOM_ASSERT(normal == Vec_t(0, 0, 1));
      VECGEOM_ASSERT(valid = cone.Normal(Vec_t(x, y, -zcut), normal));
      VECGEOM_ASSERT(normal == Vec_t(0, 0, -1));
    }
  }

  // points on top edge
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(a * (h - zcut), 0, zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, 1) + Vec_t(h, 0, a * h).Unit()).Unit());
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(a * (zcut - h), 0, zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, 1) + Vec_t(-h, 0, a * h).Unit()).Unit());
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, b * (h - zcut), zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, 1) + Vec_t(0, h, b * h).Unit()).Unit());
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, b * (zcut - h), zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, 1) + Vec_t(0, -h, b * h).Unit()).Unit());

  // points on bottom edge
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(a * (h + zcut), 0, -zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, -1) + Vec_t(h, 0, a * h).Unit()).Unit());
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(-a * (h + zcut), 0, -zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, -1) + Vec_t(-h, 0, a * h).Unit()).Unit());
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, b * (h + zcut), -zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, -1) + Vec_t(0, h, b * h).Unit()).Unit());
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, -b * (h + zcut), -zcut), normal));
  VECGEOM_ASSERT(normal == (Vec_t(0, 0, -1) + Vec_t(0, -h, b * h).Unit()).Unit());

  // points on z-axis, not on surface
  VECGEOM_ASSERT((valid = cone.Normal(Vec_t(0, 0, h), normal)) == false);
  VECGEOM_ASSERT(normal == Vec_t(0, 0, 1));
  VECGEOM_ASSERT((valid = cone.Normal(Vec_t(0, 0, 0), normal)) == false);
  VECGEOM_ASSERT(normal == Vec_t(0, 0, 1));
  VECGEOM_ASSERT((valid = cone.Normal(Vec_t(0, 0, -zcut + kTolerance), normal)) == false);
  VECGEOM_ASSERT(normal == Vec_t(0, 0, -1));
  VECGEOM_ASSERT((valid = cone.Normal(Vec_t(0, 0, -zcut - kTolerance), normal)) == false);
  VECGEOM_ASSERT(normal == Vec_t(0, 0, -1));

  // Full cone, point in apex
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 20.);
  VECGEOM_ASSERT(valid = cone.Normal(Vec_t(0, 0, h), normal));
  VECGEOM_ASSERT(normal == Vec_t(0, 0, 1));

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check SafetyToIn()
  //
  std::cout << "=== Check SafetyToIn()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  VECGEOM_ASSERT(a >= b);

  // Check inside points ("wrong side")
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      for (double scale = 0.; scale < 0.99; scale += 0.1) {
        double x = scale * a * (h - z) * std::cos(phi);
        double y = scale * b * (h - z) * std::sin(phi);
        VECGEOM_ASSERT(cone.Inside(Vec_t(x, y, z)) == vecgeom::kInside);
        VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(x, y, z)) < 0.);
      }
    }
  }

  // Check points on lateral surface
  for (double dz = 0.2, z = -zcut; z < zcut + 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double x = a * (h - z) * std::cos(phi);
      double y = b * (h - z) * std::sin(phi);
      Vec_t p(x, y, z + (0.5 - RNG::Instance().uniform()) * kHalfTolerance);
      VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
      VECGEOM_ASSERT(cone.SafetyToIn(p) == 0.);
    }
  }

  // Check points on bases
  for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
    for (double scale = 0.; scale < 1.01; scale += 0.1) {
      double x1 = scale * a * (h - zcut) * std::cos(phi);
      double y1 = scale * b * (h - zcut) * std::sin(phi);
      double z1 = zcut + (0.5 - RNG::Instance().uniform()) * kHalfTolerance;
      VECGEOM_ASSERT(cone.Inside(Vec_t(x1, y1, z1)) == vecgeom::kSurface);
      VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(x1, y1, z1)) == 0.);
      double x2 = scale * a * (h + zcut) * std::cos(phi);
      double y2 = scale * b * (h + zcut) * std::sin(phi);
      double z2 = -zcut + (0.5 - RNG::Instance().uniform()) * kHalfTolerance;
      VECGEOM_ASSERT(cone.Inside(Vec_t(x2, y2, z2)) == vecgeom::kSurface);
      VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(x2, y2, z2)) == 0.);
    }
  }

  // Check outside points located between z planes
  for (double dz = 0.2, z = -zcut; z < zcut + 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      for (double scale = 1.1; scale < 20; scale += 2.3) {
        double x    = scale * a * (h - z) * std::cos(phi);
        double y    = scale * b * (h - z) * std::sin(phi);
        double hp   = std::sqrt(x * x / a / a + y * y / b / b) + z;
        double dist = (hp - h) * b / std::sqrt(1 + b * b);
        VECGEOM_ASSERT(cone.Inside(Vec_t(x, y, z)) == vecgeom::kOutside);
        VECGEOM_ASSERT(dist > kHalfTolerance);
        VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(x, y, z)) > dist - kHalfTolerance);
      }
    }
  }

  // Check some other outside points
  VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(0, 0, h)) == h - zcut);
  VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(a * h, 0, h)) == h - zcut);
  VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(a * h, 0, -h)) == h - zcut);
  VECGEOM_ASSERT(cone.SafetyToIn(Vec_t(a * h, 0, 2 * h)) == 2 * h - zcut);

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check SafetyToOut()
  //
  std::cout << "=== Check SafetyToOut()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  VECGEOM_ASSERT(a >= b);

  // Check outside points ("wrong side")
  VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(0, 0, h)) == -h + zcut); // apex
  for (double dz = 0.2, z = -(zcut + dz); z <= (zcut + dz); z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double scale = 1.1;
      double x     = scale * a * (h - z) * std::cos(phi);
      double y     = scale * b * (h - z) * std::sin(phi);
      VECGEOM_ASSERT(cone.Inside(Vec_t(x, y, z)) == vecgeom::kOutside);
      VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(x, y, z)) < 0.);
    }
  }

  // Check points on lateral surface
  for (double dz = 0.2, z = -zcut; z < zcut + 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double x = a * (h - z) * std::cos(phi);
      double y = b * (h - z) * std::sin(phi);
      Vec_t p(x, y, z + (0.5 - RNG::Instance().uniform()) * kHalfTolerance);
      VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
      VECGEOM_ASSERT(cone.SafetyToOut(p) == 0.);
    }
  }

  // Check points on bases
  for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
    for (double scale = 0.; scale < 1.01; scale += 0.1) {
      double x1 = scale * a * (h - zcut) * std::cos(phi);
      double y1 = scale * b * (h - zcut) * std::sin(phi);
      double z1 = zcut + (0.5 - RNG::Instance().uniform()) * kHalfTolerance;
      VECGEOM_ASSERT(cone.Inside(Vec_t(x1, y1, z1)) == vecgeom::kSurface);
      VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(x1, y1, z1)) == 0.);
      double x2 = scale * a * (h + zcut) * std::cos(phi);
      double y2 = scale * b * (h + zcut) * std::sin(phi);
      double z2 = -zcut + (0.5 - RNG::Instance().uniform()) * kHalfTolerance;
      VECGEOM_ASSERT(cone.Inside(Vec_t(x2, y2, z2)) == vecgeom::kSurface);
      VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(x2, y2, z2)) == 0.);
    }
  }

  // Check inside points placed near lateral surface
  for (double dz = 0.2, z = -zcut + 0.9; z < zcut - 0.9; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double scale = 0.9;
      double x     = scale * a * (h - z) * std::cos(phi);
      double y     = scale * b * (h - z) * std::sin(phi);
      double hp    = std::sqrt(x * x / a / a + y * y / b / b) + z;
      double dist  = (h - hp) * b / std::sqrt(1 + b * b);
      VECGEOM_ASSERT(cone.Inside(Vec_t(x, y, z)) == vecgeom::kInside);
      VECGEOM_ASSERT(dist > kHalfTolerance);
      VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(x, y, z)) > dist - kHalfTolerance);
    }
  }

  // Check some other inside points
  VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(0, 0, 0)) == h * b / std::sqrt(1 + b * b));
  VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(0, 0, 0.5 * zcut)) == 0.5 * zcut);
  VECGEOM_ASSERT(cone.SafetyToOut(Vec_t(0, 0, -0.5 * zcut)) == 0.5 * zcut);

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check DistanceToIn()
  //
  std::cout << "=== Check DistanceToIn()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  VECGEOM_ASSERT(a >= b);

  // Check inside points (negative distances)
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      for (double scale = 0.; scale < 0.99; scale += 0.1) {
        double x = scale * a * (h - z) * std::cos(phi);
        double y = scale * b * (h - z) * std::sin(phi);
        Vec_t p(x, y, z);
        Vec_t v = Vec_t(x, y, z).Unit();
        VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kInside);
        VECGEOM_ASSERT(cone.DistanceToIn(p, v) < 0);
      }
    }
  }

  // Check points on lateral surface (1)
  // - small distances of different sign if point is moving to inside
  // - infinity if point is moving to outside
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double x = a * (h - z) * std::cos(phi);
      double y = b * (h - z) * std::sin(phi);
      Vec_t p(x, y, z + (0.5 - RNG::Instance().uniform()) * kHalfTolerance);
      VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
      // point is moving to outside
      Vec_t v = Vec_t(x, y, z).Unit();
      VECGEOM_ASSERT(cone.DistanceToIn(p, v) == kInfLength);
      // point is moving to inside
      double dist = cone.DistanceToIn(p, -v);
      VECGEOM_ASSERT(dist != 0 && std::abs(dist) < kHalfTolerance);
    }
  }
  // Check points on lateral surface (2)
  // - infinity if point is moving along the surface
  // - infinity if touch
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    Vec_t p(a * (h - z), 0, z);
    VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
    VECGEOM_ASSERT(valid = cone.Normal(p, normal));
    VECGEOM_ASSERT(normal == Vec_t(h, 0, a * h).Unit());
    // move along surface
    Vec_t v = Vec_t(a * h, 0, -h).Unit();
    VECGEOM_ASSERT(cone.DistanceToIn(p, v) == kInfLength);
    VECGEOM_ASSERT(cone.DistanceToIn(p, -v) == kInfLength);
    // touch
    v = normal.Cross(Vec_t(0, 0, 1)).Unit();
    VECGEOM_ASSERT(cone.DistanceToIn(p, v) == kInfLength);
    VECGEOM_ASSERT(cone.DistanceToIn(p, -v) == kInfLength);
  }

  // Check outside points
  // - infinity if point is moving parallel to the surface
  Vec_t pzax(0, 0, h + 1);                 // point is on z axis
  Vec_t vzax = Vec_t(a * h, 0, -h).Unit(); // moving parallel to x-surface (y = 0)
  VECGEOM_ASSERT(cone.Inside(pzax) == vecgeom::kOutside);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, -vzax) == kInfLength);

  pzax.Set(kTolerance, 0, h + 1); // point is inside upper nappe
  VECGEOM_ASSERT(cone.Inside(pzax) == vecgeom::kOutside);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, -vzax) == kInfLength);

  pzax.Set(-kTolerance, 0, h + 1); // point is inside upper nappe
  VECGEOM_ASSERT(cone.Inside(pzax) == vecgeom::kOutside);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, -vzax) == kInfLength);

  pzax.Set(0, kTolerance, h + 1); // point is inside upper nappe
  VECGEOM_ASSERT(cone.Inside(pzax) == vecgeom::kOutside);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, -vzax) == kInfLength);

  pzax.Set(0, -kTolerance, h + 1); // point is inside upper nappe
  VECGEOM_ASSERT(cone.Inside(pzax) == vecgeom::kOutside);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, -vzax) == kInfLength);

  pzax.Set(-1, 0, h + 1); // point is outside upper nappe
  VECGEOM_ASSERT(cone.Inside(pzax) == vecgeom::kOutside);
  VECGEOM_ASSERT(std::abs(cone.DistanceToIn(pzax, vzax) - std::sqrt(6 * 6 + 3 * 3)) < kHalfTolerance);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, -vzax) == kInfLength);

  // Special cases to check:
  //   0) Point is leaving the solid (already checked)
  //   1) Trajectory traverses the apex
  //   2) Trajectory is parallel to the surface (already checked)
  //   3) Touch / Scratching

  // Trajectory traverses the apex
  pzax.Set(0, 0, h + 1);
  vzax.Set(0, 0, -1);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == (h + 1 - zcut));

  pzax.Set(1, 0, h + 1);
  vzax = Vec_t(-1, 0, -1).Unit();
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);

  pzax.Set(1, 0, h);
  vzax.Set(0, 0, -1);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == (h - zcut));

  // Touch
  pzax.Set(7.5, 0, h);
  vzax.Set(0, 0, -1);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);

  pzax.Set(7.5 - 0.1 * kHalfTolerance, 0, h);
  vzax.Set(0, 0, -1);
  VECGEOM_ASSERT(cone.DistanceToIn(pzax, vzax) == kInfLength);

  // Check set of points at certain Z
  double Z;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  CheckDistanceToIn(cone, Z = -7.5);
  CheckDistanceToIn(cone, Z = -5 - 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = -5);
  CheckDistanceToIn(cone, Z = -5 + 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 2.5);
  CheckDistanceToIn(cone, Z = 5 - 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 5);
  CheckDistanceToIn(cone, Z = 5 + 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 7.5);
  CheckDistanceToIn(cone, Z = 10 - 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 10);
  CheckDistanceToIn(cone, Z = 10 + 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 12.5);
  CheckDistanceToIn(cone, Z = 30000.);
  CheckDistanceToIn(cone, Z = -30000.);

  cone.SetParameters(a = 1.3, b = 1.2, h = 5, zcut = 5);
  CheckDistanceToIn(cone, Z = -7.5);
  CheckDistanceToIn(cone, Z = -5 - 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = -5);
  CheckDistanceToIn(cone, Z = -5 + 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 2.5);
  CheckDistanceToIn(cone, Z = 5 - 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 5);
  CheckDistanceToIn(cone, Z = 5 + 0.9 * kHalfTolerance);
  CheckDistanceToIn(cone, Z = 10);
  CheckDistanceToIn(cone, Z = 10000.);
  CheckDistanceToIn(cone, Z = -10000.);

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check DistanceToOut()
  //
  std::cout << "=== Check DistanceToOut()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  VECGEOM_ASSERT(a >= b);

  // Check outside points ("wrong side")
  for (double dz = 0.2, z = -(zcut + dz); z <= (zcut + dz); z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double scale = 1.1;
      double x     = scale * a * (h - z) * std::cos(phi);
      double y     = scale * b * (h - z) * std::sin(phi);
      Vec_t p(x, y, z);
      Vec_t v = Vec_t(x, y, z).Unit();
      VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kOutside);
      VECGEOM_ASSERT(cone.DistanceToOut(p, v) < 0);
      VECGEOM_ASSERT(cone.DistanceToOut(p, -v) < 0);
    }
  }

  // Check points on lateral surface (1)
  // - small distances of different sign if point is moving to outside
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    for (double dphi = 10. * deg, phi = 0.; phi < kTwoPi; phi += dphi) {
      double x = a * (h - z) * std::cos(phi);
      double y = b * (h - z) * std::sin(phi);
      Vec_t p(x, y, z + (0.5 - RNG::Instance().uniform()) * kHalfTolerance);
      VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
      // point is moving to outside
      Vec_t v     = Vec_t(x, y, z).Unit();
      double dist = cone.DistanceToOut(p, v);
      VECGEOM_ASSERT(dist != 0 && std::abs(dist) < kHalfTolerance);
    }
  }

  // Check points on lateral surface (2)
  // - 0 if touch
  for (double dz = 0.2, z = -zcut + dz; z < zcut - 0.01; z += dz) {
    Vec_t p(a * (h - z), 0, z);
    VECGEOM_ASSERT(cone.Inside(p) == vecgeom::kSurface);
    VECGEOM_ASSERT(valid = cone.Normal(p, normal));
    VECGEOM_ASSERT(normal == Vec_t(1, 0, a).Unit());
    // move along surface
    Vec_t v = Vec_t(a, 0, -1).Unit();
    VECGEOM_ASSERT(cone.DistanceToOut(p, v) == 0);
    VECGEOM_ASSERT(cone.DistanceToOut(p, -v) == 0);
    // touch
    v = normal.Cross(Vec_t(0, 0, 1)).Unit();
    VECGEOM_ASSERT(cone.DistanceToOut(p, v) == 0);
    VECGEOM_ASSERT(cone.DistanceToOut(p, -v) == 0);
  }

  // Check set of points at certain Z
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  CheckDistanceToOut(cone, Z = -7.5);
  CheckDistanceToOut(cone, Z = -5 - 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = -5);
  CheckDistanceToOut(cone, Z = -5 + 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 2.5);
  CheckDistanceToOut(cone, Z = 5 - 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 5);
  CheckDistanceToOut(cone, Z = 5 + 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 7.5);
  CheckDistanceToOut(cone, Z = 10 - 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 10);
  CheckDistanceToOut(cone, Z = 10 + 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 12.5);

  cone.SetParameters(a = 1.3, b = 1.2, h = 5, zcut = 5);
  CheckDistanceToOut(cone, Z = -7.5);
  CheckDistanceToOut(cone, Z = -5 - 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = -5);
  CheckDistanceToOut(cone, Z = -5 + 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 2.5);
  CheckDistanceToOut(cone, Z = 5 - 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 5);
  CheckDistanceToOut(cone, Z = 5 + 0.9 * kHalfTolerance);
  CheckDistanceToOut(cone, Z = 10);

  ///////////////////////////////////////////////////////////////////////////////
  //
  // Check SamplePointOnSurface()
  //
  std::cout << "=== Check SamplePointOnSurface()" << std::endl;
  cone.SetParameters(a = 0.5, b = 0.4, h = 10., zcut = 5.);
  area         = cone.SurfaceArea();
  double hzpos = h - zcut;
  double hzneg = h + zcut;
  double szpos = kPi * a * b * hzpos * hzpos;
  double szneg = kPi * a * b * hzneg * hzneg;
  double sside = area - szpos - szneg;

  int nzneg = 0, nzpos = 0, nside = 0, nfactor = 10000, ntot = 4 * area * nfactor;
  for (int i = 0; i < ntot; i++) {
    Vec_t rndPoint = cone.GetUnplacedVolume()->SamplePointOnSurface();
    VECGEOM_ASSERT(cone.Inside(rndPoint) == vecgeom::kSurface);
    if (rndPoint.x() < 0 || rndPoint.y() < 0) continue;
    if (rndPoint.z() == -zcut)
      ++nzneg;
    else if (rndPoint.z() == zcut)
      ++nzpos;
    else
      ++nside;
  }
  std::cout << std::endl;
  std::cout << "szneg,sside,szpos = " << szneg << ", \t" << sside << ", \t" << szpos << std::endl;
  std::cout << "nzneg,nside,nzpos = " << nzneg << ", \t" << nside << ", \t" << nzpos << std::endl;
  VECGEOM_ASSERT(std::abs(nzneg - szneg * nfactor) < 2. * std::sqrt(ntot));
  VECGEOM_ASSERT(std::abs(nside - sside * nfactor) < 2. * std::sqrt(ntot));
  VECGEOM_ASSERT(std::abs(nzpos - szpos * nfactor) < 2. * std::sqrt(ntot));

  return true;
}

int main()
{
  VECGEOM_ASSERT(TestEllipticalCone<SimpleEllipticalCone>());
  std::cout << "\n   Test EllipticalCone passed\n" << std::endl;

  return 0;
}
