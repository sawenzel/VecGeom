#ifndef VECGEOM_SURFACE_EQUATIONS_H_
#define VECGEOM_SURFACE_EQUATIONS_H_

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/base/Math.h>

#include <VecCore/VecCore>
#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

using namespace vecCore::math;

template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

///< Second order equation coefficients, as in: <x * x + 2*phalf * x + q = 0>
template <typename Real_t>
struct QuadraticCoef {
  Real_t phalf{0}; // we don't need p itself when solving the equation, hence we store just p/2
  Real_t q{0};     // we do need q
};

///< Fourth order equation coefficients, as in: <x^4 + a*x^3 + b*x^2 + c*x + d = 0>
template <typename Real_t>
struct QuarticCoef {
  Real_t a{0};
  Real_t b{0};
  Real_t c{0};
  Real_t d{0};
};

/// @brief Fill equation for ray-cylinder intersections
/// @param point Starting point in local coordinates
/// @param dir Direction in local coordinates
/// @param radius Cylinder radius
/// @return coef Quadratic equation giving ray-cylinder intersections
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void CylinderEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Real_t radius,
                                        QuadraticCoef<Real_t> &coef)
{
  // TODO: The cylyndrical and cone equations can be merged
  Real_t rsq    = point.Perp2();
  Real_t rdotn  = point.x() * dir.x() + point.y() * dir.y();
  Real_t invnsq = static_cast<Real_t>(1.) / vecgeom::NonZero(dir.Perp2());

  coef.phalf = invnsq * rdotn;
  coef.q     = invnsq * (rsq - radius * radius);
}

/// @brief Fill equation for ray-cone intersections
/// @param point Starting point in local coordinates
/// @param dir Direction in local coordinates
/// @param radius Cone radius at Z = 0
/// @param slope Cone surface slope
/// @return coef Quadratic equation giving ray-cone intersections
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void ConeEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Real_t radius,
                                    Real_t slope, QuadraticCoef<Real_t> &coef)
{
  Real_t rz     = radius + point[2] * slope;
  Real_t rsq    = point.Perp2();
  Real_t rdotn  = point.x() * dir.x() + point.y() * dir.y() - slope * rz * dir.z();
  Real_t invnsq = static_cast<Real_t>(1.) / vecgeom::NonZero(dir.Perp2() - slope * slope * dir.z() * dir.z());

  coef.phalf = invnsq * rdotn;
  coef.q     = invnsq * (rsq - rz * rz);
}

/// @brief Fill equation for ray-sphere intersections
/// @param point Starting point in local coordinates
/// @param dir Direction in local coordinates
/// @param radius Sphere radius
/// @return coef Quadratic equation giving ray-sphere intersections
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void SphereEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Real_t radius,
                                      QuadraticCoef<Real_t> &coef)
{
  Real_t rsq = point.Mag2();
  coef.phalf = point.Dot(dir);
  coef.q     = rsq - radius * radius;
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void TorusEq(Vector3D<Real_t> const &pt, Vector3D<Real_t> const &dir, const Real_t radius_tor,
                                     const Real_t radius_tube, QuarticCoef<Real_t> &coef)
{
  // to be taken from ROOT
  // Returns distance to the surface or the torus from a point, along
  // a direction. Point is close enough to the boundary so that the distance
  // to the torus is decreasing while moving along the given direction.

  // Compute coeficients of the quartic
  Real_t radius_tor2  = radius_tor * radius_tor; // these could be precalculated
  Real_t radius_tube2 = radius_tube * radius_tube;

  Real_t r0sq   = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
  Real_t rdotn  = pt[0] * dir[0] + pt[1] * dir[1] + pt[2] * dir[2];
  Real_t rsumsq = radius_tor2 + radius_tube2;
  coef.a        = 4. * rdotn;
  coef.b        = 2. * (r0sq + 2. * rdotn * rdotn - rsumsq + 2. * radius_tor2 * dir[2] * dir[2]);
  coef.c        = 4. * (r0sq * rdotn - rsumsq * rdotn + 2. * radius_tor2 * pt[2] * dir[2]);
  coef.d        = r0sq * r0sq - 2. * r0sq * rsumsq + 4. * radius_tor2 * pt[2] * pt[2] +
           (radius_tor2 - radius_tube2) * (radius_tor2 - radius_tube2);
};

/// @brief Solver for quadratic equations
/// @tparam Real_t Floating point type
/// @param coef Quadratic equation coefficients
/// @param roots Equation roots
/// @param numroots Number of roots grater than -kTolerance
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void QuadraticSolver(QuadraticCoef<Real_t> const &coef, Real_t *roots, int &numroots)
{
  numroots     = 0;
  Real_t delta = coef.phalf * coef.phalf - coef.q;
  if (delta < Real_t(0)) return;
  Real_t r1 = -coef.phalf - Sign(coef.phalf) * Sqrt(delta);
  Real_t r2 = coef.q / vecgeom::NonZero(r1);
  numroots += int(r1 > -vecgeom::kToleranceStrict<Real_t>) + int(r2 > -vecgeom::kToleranceStrict<Real_t>);
  roots[0] = Min(r1, r2);
  roots[1] = Max(r1, r2);
  if (numroots == 1) roots[0] = roots[1];
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE unsigned int SolveCubic(Real_t a, Real_t b, Real_t c, Real_t *x)
{
  // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
  // Input: a,b,c
  // Output: x[3] real solutions
  // Returns number of real solutions (1 or 3)
  const Real_t ott     = 1. / 3.;
  const Real_t sq3     = Sqrt(3.);
  const Real_t inv6sq3 = 1. / (6. * sq3);
  unsigned int ireal   = 1;
  Real_t p             = b - a * a * ott;
  Real_t q             = c - a * b * ott + 2. * a * a * a * ott * ott * ott;
  Real_t delta         = 4 * p * p * p + 27. * q * q;
  Real_t t, u;

  if (delta >= 0) {
    delta = Sqrt(delta);
    t     = (-3 * q * sq3 + delta) * inv6sq3;
    u     = (3 * q * sq3 + delta) * inv6sq3;
    x[0]  = CopySign(Real_t(1.), t) * Cbrt(Abs(t)) - CopySign(Real_t(1.), u) * Cbrt(Abs(u)) - a * ott;
  } else {
    delta = Sqrt(-delta);
    t     = -0.5 * q;
    u     = delta * inv6sq3;
    x[0]  = 2. * Pow(t * t + u * u, Real_t(0.5) * ott) * cos(ott * ATan2(u, t));
    x[0] -= a * ott;
  }

  t     = x[0] * x[0] + a * x[0] + b;
  u     = a + x[0];
  delta = u * u - Real_t(4.) * t;
  if (delta >= 0) {
    ireal = 3;
    delta = Sqrt(delta);
    x[1]  = Real_t(0.5) * (-u - delta);
    x[2]  = Real_t(0.5) * (-u + delta);
  }

  return ireal;
}

template <typename Real_t, unsigned int i, unsigned int j>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CmpAndSwap(Real_t *array)
{
  if (array[i] > array[j]) {
    Real_t c = array[j];
    array[j] = array[i];
    array[i] = c;
  }
}

// a special function to sort a 4 element array
// sorting is done inplace and in increasing order
// implementation comes from a sorting network
template <typename Real_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Sort4(Real_t *array)
{
  CmpAndSwap<Real_t, 0, 2>(array);
  CmpAndSwap<Real_t, 1, 3>(array);
  CmpAndSwap<Real_t, 0, 1>(array);
  CmpAndSwap<Real_t, 2, 3>(array);
  CmpAndSwap<Real_t, 1, 2>(array);
}

// solve quartic taken from ROOT/TGeo and adapted
//_____________________________________________________________________________

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void QuarticSolver(QuarticCoef<Real_t> const &coef, Real_t *roots, int &numroots)
{
  // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
  // Input: coefficients a,b,c,d
  // Output: roots[4] - real solutions
  //         numroots number of real solutions (0 to 3)
  Real_t e = coef.b - 3. * coef.a * coef.a / 8.;
  Real_t f = coef.c + coef.a * coef.a * coef.a / 8. - 0.5 * coef.a * coef.b;
  Real_t g =
      coef.d - 3. * coef.a * coef.a * coef.a * coef.a / 256. + coef.a * coef.a * coef.b / 16. - coef.a * coef.c / 4.;
  Real_t xx[4] = {vecgeom::InfinityLength<Real_t>(), vecgeom::InfinityLength<Real_t>(),
                  vecgeom::InfinityLength<Real_t>(), vecgeom::InfinityLength<Real_t>()};
  Real_t delta;
  Real_t h = 0.;
  numroots = 0;

  // special case when f is zero
  if (Abs(f) < 1E-6) {
    delta = e * e - 4. * g;
    if (delta < 0.) return;
    delta = Sqrt(delta);
    h     = 0.5 * (-e - delta);
    if (h >= 0) {
      h                 = Sqrt(h);
      roots[numroots++] = -h - 0.25 * coef.a;
      roots[numroots++] = h - 0.25 * coef.a;
    }
    h = 0.5 * (-e + delta);
    if (h >= 0) {
      h                 = Sqrt(h);
      roots[numroots++] = -h - 0.25 * coef.a;
      roots[numroots++] = h - 0.25 * coef.a;
    }
    Sort4(roots);
    return;
  }

  if (Abs(g) < 1E-6) {
    roots[numroots++] = -0.25 * coef.a;
    // this actually wants to solve a second order equation
    // we should specialize if it happens often
    unsigned int ncubicroots = SolveCubic<Real_t>(0, e, f, xx);
    // this loop is not nice
    for (unsigned int i = 0; i < ncubicroots; i++)
      roots[numroots++] = xx[i] - 0.25 * coef.a;
    Sort4(roots); // could be Sort3
    return;
  }

  numroots = SolveCubic<Real_t>(2. * e, e * e - 4. * g, -f * f, xx);
  if (numroots == 1) {
    if (xx[0] <= 0) return;
    h = Sqrt(xx[0]);
  } else {
    // 3 real solutions of the cubic
    for (unsigned int i = 0; i < 3; i++) {
      h = xx[i];
      if (h >= 0) break;
    }
    if (h <= 0) return;
    h = Sqrt(h);
  }
  Real_t j = 0.5 * (e + h * h - f / h);
  numroots = 0;
  delta    = h * h - 4. * j;
  if (delta >= 0) {
    delta             = Sqrt(delta);
    roots[numroots++] = 0.5 * (-h - delta) - 0.25 * coef.a;
    roots[numroots++] = 0.5 * (-h + delta) - 0.25 * coef.a;
  }
  delta = h * h - 4. * g / j;
  if (delta >= 0) {
    delta             = Sqrt(delta);
    roots[numroots++] = 0.5 * (h - delta) - 0.25 * coef.a;
    roots[numroots++] = 0.5 * (h + delta) - 0.25 * coef.a;
  }
  Sort4(roots);
  return;
}

} // namespace vgbrep
#endif
