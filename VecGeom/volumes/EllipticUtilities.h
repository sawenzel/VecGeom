// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Collection of auxiliary utilities for elliptical shapes
/// @file volumes/EllipticUtilities.h
/// @author Original implementation by Evgueni Tcherniaev (evc)

#ifndef VECGEOM_ELLIPTICUTILITIES_H_
#define VECGEOM_ELLIPTICUTILITIES_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Vector2D.h"
#include "VecGeom/base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace EllipticUtilities {

/// Computes the Complete Elliptical Integral of the Second Kind
/// @param e eccentricity
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Precision comp_ellint_2(Precision e)
{
  // The algorithm is based upon Carlson B.C., "Computation of real
  // or complex elliptic integrals", Numerical Algorithms,
  // Volume 10, Issue 1, 1995 (see equations 2.36 - 2.39)
  //
  const Precision eps = 1. / 134217728.; // 1 / 2^27

  Precision a = 1.;
  Precision b = vecCore::math::Sqrt((1. - e) * (1. + e));
  if (b == 1.) return kHalfPi;
  if (b == 0.) return 1.;

  Precision x = 1.;
  Precision y = b;
  Precision S = 0.;
  Precision M = 1.;
  while (x - y > eps * y) {
    Precision tmp = (x + y) * 0.5;

    y = vecCore::math::Sqrt(x * y);
    x = tmp;
    M += M;
    S += M * (x - y) * (x - y);
  }
  return 0.5 * kHalfPi * ((a + b) * (a + b) - S) / (x + y);
}

/// Computes perimeter of ellipse (X/A)^2 + (Y/B)^2 = 1
/// @param A X semi-axis
/// @param B Y semi-axis
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Precision EllipsePerimeter(Precision A, Precision B)
{
  Precision dx  = vecCore::math::Abs(A);
  Precision dy  = vecCore::math::Abs(B);
  Precision a   = vecCore::math::Max(dx, dy);
  Precision b   = vecCore::math::Min(dx, dy);
  Precision b_a = b / a;
  Precision e   = vecCore::math::Sqrt((1. - b_a) * (1. + b_a));
  return 4. * a * comp_ellint_2(e);
}

/// Computes the lateral surface area of elliptic cone
/// @param pA X semi-axis of ellipse at base (Z = 0)
/// @param pB Y semi-axis of ellipse at base (Z = 0)
/// @param pH height
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Precision EllipticalConeLateralArea(Precision pA, Precision pB, Precision pH)
{
  Precision x  = vecCore::math::Abs(pA);
  Precision y  = vecCore::math::Abs(pB);
  Precision h  = vecCore::math::Abs(pH);
  Precision a  = vecCore::math::Max(x, y);
  Precision b  = vecCore::math::Min(x, y);
  Precision aa = a * a;
  Precision bb = b * b;
  Precision hh = h * h;
  Precision e  = vecCore::math::Sqrt(((a - b) * (a + b) * hh) / ((hh + bb) * aa));
  return 2. * a * vecCore::math::Sqrt(hh + bb) * comp_ellint_2(e);
}

/// Picks random point inside ellipse (x/a)^2 + (y/b)^2 = 1
/// @param a X semi-axis
/// @param b Y semi-axis
//
inline Vector2D<Precision> RandomPointInEllipse(Precision a, Precision b)
{
  Precision aa = (a * a == 0.) ? 0. : 1. / (a * a);
  Precision bb = (b * b == 0.) ? 0. : 1. / (b * b);
  for (int i = 0; i < 1000; ++i) {
    Precision x = a * (2. * RNG::Instance().uniform() - 1.);
    Precision y = b * (2. * RNG::Instance().uniform() - 1.);
    if (x * x * aa + y * y * bb <= 1.) return Vector2D<Precision>(x, y);
  }
  return Vector2D<Precision>(0., 0.);
}

/// Picks random point on ellipse (x/a)^2 + (y/b)^2 = 1
/// @param a X semi-axis
/// @param b Y semi-axis
//
inline Vector2D<Precision> RandomPointOnEllipse(Precision a, Precision b)
{
  Precision A      = vecCore::math::Abs(a);
  Precision B      = vecCore::math::Abs(b);
  Precision mu_max = vecCore::math::Max(A, B);

  Precision x, y;
  for (int i = 0; i < 1000; ++i) {
    Precision phi = kTwoPi * RNG::Instance().uniform();
    x             = vecCore::math::Cos(phi);
    y             = vecCore::math::Sin(phi);
    Precision Bx  = B * x;
    Precision Ay  = A * y;
    Precision mu  = vecCore::math::Sqrt(Bx * Bx + Ay * Ay);
    if (mu_max * RNG::Instance().uniform() <= mu) break;
  }
  return Vector2D<Precision>(A * x, B * y);
}

} // namespace EllipticUtilities
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_ELLIPTICUTILITIES_H_ */
