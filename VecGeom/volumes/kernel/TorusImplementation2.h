/// @file TorusImplementation2.h
/// @brief Torus kernel helpers and navigation entry points.

#ifndef VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION2_H_
#define VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION2_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/TubeImplementation.h"
#include "VecGeom/volumes/TorusStruct2.h"

#include <cstdio>
#include <VecCore/VecCore>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TorusImplementation2;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TorusImplementation2);

inline namespace VECGEOM_IMPL_NAMESPACE {

//_____________________________________________________________________________
/// @brief Solve a monic cubic equation and keep its real roots.
/// @details Solves `x^3 + a*x^2 + b*x + c = 0`. The returned roots are used by
/// the quartic solver for torus surface intersections.
/// @param a Quadratic coefficient.
/// @param b Linear coefficient.
/// @param c Constant coefficient.
/// @param[out] x Storage for up to three real roots.
/// @return Number of real roots stored in @p x.
template <typename T>
VECCORE_ATT_HOST_DEVICE unsigned int SolveCubic(T a, T b, T c, T *x)
{
  // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
  // Input: a,b,c
  // Output: x[3] real solutions
  // Returns number of real solutions (1 or 3)
  const T ott        = 1. / 3.;
  const T sq3        = Sqrt(3.);
  const T inv6sq3    = 1. / (6. * sq3);
  unsigned int ireal = 1;
  T p                = b - a * a * ott;
  T q                = c - a * b * ott + 2. * a * a * a * ott * ott * ott;
  T delta            = 4 * p * p * p + 27. * q * q;
  T t, u;

  if (delta >= 0) {
    delta = Sqrt(delta);
    t     = (-3 * q * sq3 + delta) * inv6sq3;
    u     = (3 * q * sq3 + delta) * inv6sq3;
    x[0]  = CopySign(T(1.), t) * Cbrt(Abs(t)) - CopySign(T(1.), u) * Cbrt(Abs(u)) - a * ott;
  } else {
    delta = Sqrt(-delta);
    t     = -0.5 * q;
    u     = delta * inv6sq3;
    x[0]  = 2. * Pow(t * t + u * u, T(0.5) * ott) * cos(ott * ATan2(u, t));
    x[0] -= a * ott;
  }

  t     = x[0] * x[0] + a * x[0] + b;
  u     = a + x[0];
  delta = u * u - T(4.) * t;
  if (delta >= 0) {
    ireal = 3;
    delta = Sqrt(delta);
    x[1]  = T(0.5) * (-u - delta);
    x[2]  = T(0.5) * (-u + delta);
  }

  return ireal;
}

/// @brief Compare two fixed entries and swap them into increasing order.
/// @param[in,out] array Small root array to update in place.
template <typename T, unsigned int i, unsigned int j>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CmpAndSwap(T *array)
{
  if (array[i] > array[j]) {
    T c      = array[j];
    array[j] = array[i];
    array[i] = c;
  }
}

/// @brief Sort four candidate roots in place.
/// @details Uses a fixed sorting network to avoid branches and loops whose
/// overhead would dominate this tiny array.
/// @param[in,out] array Four entries sorted into increasing order.
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Sort4(T *array)
{
  CmpAndSwap<T, 0, 2>(array);
  CmpAndSwap<T, 1, 3>(array);
  CmpAndSwap<T, 0, 1>(array);
  CmpAndSwap<T, 2, 3>(array);
  CmpAndSwap<T, 1, 2>(array);
}

//_____________________________________________________________________________

/// @brief Solve a monic quartic equation and keep its real roots.
/// @details Solves `x^4 + a*x^3 + b*x^2 + c*x + d = 0`. Torus distance
/// kernels use these roots as candidate intersections with a scaled torus
/// surface and perform geometry-specific filtering afterwards.
/// @param a Cubic coefficient.
/// @param b Quadratic coefficient.
/// @param c Linear coefficient.
/// @param d Constant coefficient.
/// @param[out] x Storage for up to four real roots, sorted when returned.
/// @return Number of real roots stored in @p x.
template <typename T>
VECCORE_ATT_HOST_DEVICE int SolveQuartic(T a, T b, T c, T d, T *x)
{
  // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
  // Input: a,b,c,d
  // Output: x[4] - real solutions
  // Returns number of real solutions (0 to 3)
  T e     = b - 3. * a * a / 8.;
  T f     = c + a * a * a / 8. - 0.5 * a * b;
  T g     = d - 3. * a * a * a * a / 256. + a * a * b / 16. - a * c / 4.;
  T xx[4] = {vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength};
  T delta;
  T h                = 0.;
  unsigned int ireal = 0;

  // special case when f is zero
  if (Abs(f) < T(1.e-5)) {
    delta = e * e - 4. * g;
    if (delta < 0.) return 0;
    delta = Sqrt(delta);
    h     = 0.5 * (-e - delta);
    if (h >= 0) {
      h          = Sqrt(h);
      x[ireal++] = -h - 0.25 * a;
      x[ireal++] = h - 0.25 * a;
    }
    h = 0.5 * (-e + delta);
    if (h >= 0) {
      h          = Sqrt(h);
      x[ireal++] = -h - 0.25 * a;
      x[ireal++] = h - 0.25 * a;
    }
    Sort4(x);
    return ireal;
  }

  if (Abs(g) < 1E-6) {
    x[ireal++] = -0.25 * a;
    // this actually wants to solve a second order equation
    // we should specialize if it happens often
    unsigned int ncubicroots = SolveCubic<T>(0, e, f, xx);
    // this loop is not nice
    for (unsigned int i = 0; i < ncubicroots; i++)
      x[ireal++] = xx[i] - 0.25 * a;
    Sort4(x); // could be Sort3
    return ireal;
  }

  ireal = SolveCubic<T>(2. * e, e * e - 4. * g, -f * f, xx);
  if (ireal == 1) {
    if (xx[0] <= 0) return 0;
    h = Sqrt(xx[0]);
  } else {
    // 3 real solutions of the cubic
    for (unsigned int i = 0; i < 3; i++) {
      h = xx[i];
      if (h >= 0) break;
    }
    if (h <= 0) return 0;
    h = Sqrt(h);
  }
  T j   = 0.5 * (e + h * h - f / h);
  ireal = 0;
  delta = h * h - 4. * j;
  if (delta >= 0) {
    delta      = Sqrt(delta);
    x[ireal++] = 0.5 * (-h - delta) - 0.25 * a;
    x[ireal++] = 0.5 * (-h + delta) - 0.25 * a;
  }
  delta = h * h - 4. * g / j;
  if (delta >= 0) {
    delta      = Sqrt(delta);
    x[ireal++] = 0.5 * (h - delta) - 0.25 * a;
    x[ireal++] = 0.5 * (h + delta) - 0.25 * a;
  }
  Sort4(x);
  return ireal;
}

class PlacedTorus2;
template <typename T>
struct TorusStruct2;
class UnplacedTorus2;

class SIMDUnplacedTorus2;

/// @brief Kernel implementation for torus navigation.
/// @details The implementation supports full and phi-cut torus variants, with
/// optional inner radius for hollow tori. Distance queries scale the torus major
/// radius to one, solve the quartic surface equation, and then filter candidate
/// roots against radial, phi-wedge, direction, and tolerance conventions.
///
/// Navigation entry points are called with scalar inputs. The implementation
/// therefore favors local scalar predicates and direct access to cached wedge
/// state in hot paths instead of broader shape classification helpers.
struct TorusImplementation2 {
  using PlacedShape_t    = PlacedTorus2;
  using UnplacedStruct_t = TorusStruct2<Precision>;
  using UnplacedVolume_t = UnplacedTorus2;

  /// @brief Return the squared radial cross-section distance after propagation.
  /// @details The torus is assumed to be scaled so `rtor == 1`. The result is
  /// compared with scaled `rmin^2` and `rmax^2` to decide whether a phi-plane
  /// intersection lies in the torus annulus.
  /// @param point Scaled starting point.
  /// @param dir Direction of propagation.
  /// @param dist Candidate distance along @p dir.
  /// @return Squared distance to the center of the scaled torus tube.
  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static Real_v DistSqrToTorusR(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                                        Real_v dist)
  {
    Vector3D<Real_v> p = point + dir * dist;
    Real_v rxy         = p.Perp();
    return (rxy - 1.) * (rxy - 1.) + p.z() * p.z();
  }

  /// @brief Compute distance from an interior point to the torus boundary.
  /// @details Candidate distances are obtained from the outer tube surface,
  /// optional inner tube surface, and optional phi wedge. Phi candidates use the
  /// cached wedge directly because `GetWedge()` returns by value and copying it
  /// is measurable in sector hot paths.
  /// @param torus Torus runtime data.
  /// @param point Query point, expected to be inside or on the torus.
  /// @param dir Normalized propagation direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Distance to exit, `-1` for outside input.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &torus,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const & /*stepMax*/, Real_v &distance)
  {
    using Inside_v = vecCore::Index_v<Real_v>;

    bool hasphi  = (torus.dphi() < kTwoPi);
    bool hasrmin = (torus.rmin() > 0);
    Real_v rtor  = torus.rtor();

    //=== First, for points outside --> return infinity
    bool done = false;
    distance  = kInfLength;

    // very simple calculations -- only if can save some time
    Real_v distz = Abs(point.z()) - torus.rmax();
    done |= distz > kHalfTolerance;

    // outside of bounding tube?
    Real_v rsq = point.x() * point.x() + point.y() * point.y();
    // Real_v rdotv = point.x()*dir.x() + point.y()*dir.y();
    Precision outerExclRadius = torus.rtor() + torus.rmax() + kHalfTolerance;
    done |= rsq > outerExclRadius * outerExclRadius;
    Precision innerExclRadius = torus.rtor() - torus.rmax() - kHalfTolerance;
    done |= rsq < innerExclRadius * innerExclRadius;
    if (done) {
      distance = Real_v(-1.);
      return;
    }

    //=== Use InsideKernel() for a quick check, and if outside --> return -1
    Inside_v locus;
    TorusImplementation2::InsideKernel<Real_v, Inside_v>(torus, point, locus);
    if (locus == Inside_v(EInside::kOutside)) {
      distance = Real_v(-1.);
      return;
    }
    bool skipZeroOuterBoundary = false;
    bool skipZeroInnerBoundary = false;
    if (locus == Inside_v(EInside::kSurface)) {
      bool radialExit = IsOnRadialSurfaceAndMoving<Real_v, false, true>(torus, point, dir) ||
                        IsOnRadialSurfaceAndMovingTangentially<Real_v, false, false>(torus, point, dir) ||
                        (hasrmin && (IsOnRadialSurfaceAndMoving<Real_v, true, true>(torus, point, dir) ||
                                     IsOnRadialSurfaceAndMovingTangentially<Real_v, true, false>(torus, point, dir)));
      skipZeroOuterBoundary = IsOnRadialSurfaceAndMovingTangentially<Real_v, false, true>(torus, point, dir);
      skipZeroInnerBoundary = hasrmin && IsOnRadialSurfaceAndMovingTangentially<Real_v, true, true>(torus, point, dir);
      auto const &wedge     = torus.fPhiWedge;
      bool phiExit          = hasphi && InsideRadialCrossSection<Real_v>(torus, point) &&
                              (wedge.template IsPointOnSurfaceAndMovingOut<Real_v, true, true>(point, dir) ||
                               wedge.template IsPointOnSurfaceAndMovingOut<Real_v, false, true>(point, dir));
      if (radialExit || phiExit) {
        distance = Real_v(0.);
        return;
      }
    }

    Vector3D<Real_v> scaledPoint = point / rtor;
    Real_v distOut = skipZeroOuterBoundary
                         ? ToBoundary<Real_v, false>(torus, scaledPoint, dir, torus.rmax() / rtor, true, true)
                         : ToBoundary<Real_v, false>(torus, scaledPoint, dir, torus.rmax() / rtor, true);
    // ToBoundary<Backend, false, true>(torus, point, dir, torus.rmax());
    Real_v din(kInfLength);
    if (hasrmin) {
      din = skipZeroInnerBoundary ? ToBoundary<Real_v, true>(torus, scaledPoint, dir, torus.rmin() / rtor, true, true)
                                  : ToBoundary<Real_v, true>(torus, scaledPoint, dir, torus.rmin() / rtor, true);
      // ToBoundary<Backend, true, true>(torus, point, dir, torus.rmin());
    }
    distance = Min(distOut, din);
    distance *= rtor;
    // std::cerr << "distOut, din: " << distOut << ", " << din << '\n';
    // std::cerr << "distance = Min(distOut, din): " << distance << '\n';

    if (hasphi) {
      Real_v distPhi1;
      Real_v distPhi2;
      auto const &wedge = torus.fPhiWedge;
      wedge.DistanceToOut<Real_v>(point, dir, distPhi1, distPhi2);
      if (distPhi1 < distance) {
        Vector3D<Real_v> intersectionPoint = point + dir * distPhi1;
        bool insideDisk                    = InsideRadialCrossSection<Real_v>(torus, intersectionPoint);

        if (insideDisk) // Inside Disk
        {
          auto along1 = wedge.GetAlong1();
          Real_v diri = intersectionPoint.x() * along1.x() + intersectionPoint.y() * along1.y();
          if (diri >= Real_v(0.)) distance = distPhi1;
        }
      }
      if (distPhi2 < distance) {

        Vector3D<Real_v> intersectionPoint = point + dir * distPhi2;
        bool insideDisk                    = InsideRadialCrossSection<Real_v>(torus, intersectionPoint);
        if (insideDisk) // Inside Disk
        {
          auto along2  = wedge.GetAlong2();
          Real_v diri2 = intersectionPoint.x() * along2.x() + intersectionPoint.y() * along2.y();
          if (diri2 >= Real_v(0.)) distance = distPhi2;
        }
      }
    }

    if (distance >= kInfLength) distance = (locus == Inside_v(EInside::kSurface)) ? Real_v(0.) : Real_v(-1.);
  }

  /// @brief Classification helper used by `Contains` variants.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] inside True when @p point is accepted by the requested torus
  /// predicate.
  template <typename Real_v, bool notForDisk>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ContainsKernel(UnplacedStruct_t const &torus,
                                                                          Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused  = false;
    bool outside = false;
    TorusImplementation2::GenericKernelForContainsAndInside<Real_v, false, notForDisk>(torus, point, unused, outside);
    inside = !outside;
  }

  /// @brief Test whether a point belongs to the radial disk of a cut torus.
  /// @details This keeps the phi wedge out of the predicate, allowing callers
  /// to validate intersections with a phi boundary plane against only the
  /// torus radial annulus.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] inside True if the point lies in the torus radial annulus.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContainsDisk(UnplacedStruct_t const &torus,
                                                                                Vector3D<Real_v> const &point,
                                                                                bool &inside)
  {
    ContainsKernel<Real_v, false>(torus, point, inside);
  }

  /// @brief Test the torus radial cross-section only.
  /// @details This is intentionally narrower than full torus classification:
  /// it ignores phi because phi-plane distance checks already know they are on
  /// the limiting plane and only need to know whether material exists there.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @return True if @p point lies between `rmin` and `rmax` in the torus
  /// cross-section.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool InsideRadialCrossSection(UnplacedStruct_t const &torus,
                                                                                    Vector3D<Real_v> const &point)
  {
    Real_v rxy   = Sqrt(point[0] * point[0] + point[1] * point[1]);
    Real_v radsq = (rxy - torus.rtor()) * (rxy - torus.rtor()) + point[2] * point[2];
    return radsq <= torus.rmax2() && radsq >= torus.rmin2();
  }

  template <typename Real_v, bool ForRmin, bool MovingOut>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnRadialSurfaceAndMoving(UnplacedStruct_t const &torus,
                                                                                      Vector3D<Real_v> const &point,
                                                                                      Vector3D<Real_v> const &dir)
  {
    Real_v radius = ForRmin ? torus.rmin() : torus.rmax();
    if (radius == Real_v(0.)) return false;

    Real_v rxy2 = point.x() * point.x() + point.y() * point.y();
    if (rxy2 == Real_v(0.)) return false;

    Real_v rxy    = Sqrt(rxy2);
    Real_v dr     = rxy - torus.rtor();
    Real_v radsq  = dr * dr + point.z() * point.z();
    Real_v tolRad = Real_v(100. * vecgeom::kTolerance) * radius;
    if (Abs(radsq - radius * radius) > tolRad) return false;

    Real_v motion = dr * (point.x() * dir.x() + point.y() * dir.y()) / rxy + point.z() * dir.z();
    if (ForRmin) motion = -motion;
    return MovingOut ? motion > kToleranceDist<Real_v> : motion < -kToleranceDist<Real_v>;
  }

  template <typename Real_v, bool ForRmin, bool MovingToMaterial>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnRadialSurfaceAndMovingTangentially(
      UnplacedStruct_t const &torus, Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir)
  {
    Real_v radius = ForRmin ? torus.rmin() : torus.rmax();
    if (radius == Real_v(0.)) return false;

    Real_v rxy2 = point.x() * point.x() + point.y() * point.y();
    if (rxy2 == Real_v(0.)) return false;

    Real_v rxy    = Sqrt(rxy2);
    Real_v dr     = rxy - torus.rtor();
    Real_v radsq  = dr * dr + point.z() * point.z();
    Real_v tolRad = Real_v(100. * vecgeom::kTolerance) * radius;
    if (Abs(radsq - radius * radius) > tolRad) return false;

    Real_v radialDot   = point.x() * dir.x() + point.y() * dir.y();
    Real_v rxyDot      = radialDot / rxy;
    Real_v firstMotion = dr * rxyDot + point.z() * dir.z();
    if (Abs(firstMotion) > kToleranceDist<Real_v>) return false;

    Real_v dirPerp2        = dir.x() * dir.x() + dir.y() * dir.y();
    Real_v rxySecond       = (dirPerp2 * rxy2 - radialDot * radialDot) / (rxy2 * rxy);
    Real_v rxyDot2         = rxyDot * rxyDot;
    Real_v curvatureMotion = dr * rxySecond;
    Real_v zMotion         = dir.z() * dir.z();
    Real_v secondMotion    = rxyDot2 + curvatureMotion + zMotion;
    Real_v secondTol       = kEpsilonT<Real_v> * (Abs(rxyDot2) + Abs(curvatureMotion) + Abs(zMotion));
    // For zero first-order motion, use the local curvature of
    // q=(rxy-rtor)^2+z^2. Material lies above rmin and below rmax.
    bool materialSide = false;
    bool classified   = false;
    if (secondMotion > secondTol) {
      classified   = true;
      materialSide = ForRmin;
    } else if (secondMotion < -secondTol) {
      classified   = true;
      materialSide = !ForRmin;
    } else {
      // Neutral radial tangents split at third order except at tube apexes,
      // where azimuthal tangents move to larger q only at fourth order.
      Real_v tangentialPerp2 = dirPerp2 - rxyDot2;
      Real_v tangentialTol   = kEpsilonT<Real_v> * (Abs(dirPerp2) + Abs(rxyDot2));
      Real_v thirdSign       = rxyDot * tangentialPerp2;
      Real_v thirdTol        = Abs(rxyDot) * tangentialTol;
      if (thirdSign > thirdTol) {
        classified   = true;
        materialSide = ForRmin;
      } else if (thirdSign < -thirdTol) {
        classified   = true;
        materialSide = !ForRmin;
      } else if (Abs(dr) <= kToleranceDist<Real_v> && tangentialPerp2 > tangentialTol) {
        classified   = true;
        materialSide = ForRmin;
      }
    }
    return classified && (MovingToMaterial ? materialSide : !materialSide);
  }

  /// @brief Return inside/surface/outside classification for a point.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] inside VecGeom inside code.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void InsideKernel(UnplacedStruct_t const &torus,
                                                                        Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside  = false;
    bool completelyoutside = false;
    TorusImplementation2::GenericKernelForContainsAndInside<Real_v, true, true>(torus, point, completelyinside,
                                                                                completelyoutside);
    inside = completelyoutside ? Inside_t(EInside::kOutside)
                               : (completelyinside ? Inside_t(EInside::kInside) : Inside_t(EInside::kSurface));
  }

  /// @brief Shared point-classification kernel for `Contains` and `Inside`.
  /// @details The radial tube classification is based on the squared distance
  /// from the point to the torus tube centerline. For phi-cut tori, the cached
  /// wedge predicate is applied only when the caller requests full shape
  /// classification rather than a radial disk predicate.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] completelyinside True when @p point is strictly inside for the
  /// selected predicate.
  /// @param[out] completelyoutside True when @p point is outside for the
  /// selected predicate.
  template <typename Real_v, bool ForInside, bool notForDisk>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &torus, Vector3D<Real_v> const &point, bool &completelyinside, bool &completelyoutside)

  {
    // using vecgeom::GenericKernels;
    // here we are explicitly unrolling the loop since a for statement will likely be a penalty
    // check if second call to Abs is compiled away
    // and it can anyway not be vectorized
    /* rmax */
    VECGEOM_CONST Precision tol = 100. * vecgeom::kTolerance;

    Real_v rxy   = Sqrt(point[0] * point[0] + point[1] * point[1]);
    Real_v radsq = (rxy - torus.rtor()) * (rxy - torus.rtor()) + point[2] * point[2];

    if (ForInside) {
      completelyoutside = radsq > (tol * torus.rmax() + torus.rmax2()); // rmax
      completelyinside  = radsq < (-tol * torus.rmax() + torus.rmax2());
    } else {
      completelyoutside = radsq > torus.rmax2();
    }

    if (completelyoutside) return;
    /* rmin */
    if (ForInside) {
      completelyoutside |= radsq < (-tol * torus.rmin() + torus.rmin2()); // rmin
      completelyinside &= radsq > (tol * torus.rmin() + torus.rmin2());
    } else {
      completelyoutside |= radsq < torus.rmin2();
    }

    if (completelyoutside) return;

    /* phi */
    if ((torus.dphi() < kTwoPi) && (notForDisk)) {
      bool completelyoutsidephi = false;
      bool completelyinsidephi  = false;
      torus.fPhiWedge.template GenericKernelForContainsAndInside<ForInside>(point, completelyinsidephi,
                                                                            completelyoutsidephi);

      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  /// @brief Find the next crossing of a scaled torus tube boundary.
  /// @details The caller passes a point already scaled by `rtor` and a scaled
  /// tube radius. The method solves the quartic, filters candidate roots by
  /// propagation direction and phi wedge ownership, and refines the accepted
  /// root with Newton iterations. Grazing roots are rejected using tolerance
  /// bands rather than strict zero tests.
  /// @param torus Torus runtime data.
  /// @param pt Scaled query point.
  /// @param dir Normalized propagation direction.
  /// @param radius Scaled tube radius, either `rmax/rtor` or `rmin/rtor`.
  /// @param out True for distance-to-out filtering, false for distance-to-in.
  /// @param skipZero Ignore zero-distance roots from the current radial
  /// boundary.
  /// @return Accepted scaled distance, or `kInfLength` when no crossing applies.
  template <typename Real_v, bool ForRmin>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v ToBoundary(UnplacedStruct_t const &torus,
                                                                        Vector3D<Real_v> const &pt,
                                                                        Vector3D<Real_v> const &dir, Real_v radius,
                                                                        bool out, bool skipZero = false)
  {
    // to be taken from ROOT
    // Returns distance to the surface or the torus from a point, along
    // a direction. Point is close enough to the boundary so that the distance
    // to the torus is decreasing while moving along the given direction.

    // Compute coefficients of the quartic
    Real_v s                    = vecgeom::kInfLength;
    VECGEOM_CONST Real_v tol    = 100. * vecgeom::kTolerance;
    VECGEOM_CONST Real_v dirTol = kToleranceDist<Real_v>;
    Real_v r0sq                 = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
    Real_v rdotn                = pt[0] * dir[0] + pt[1] * dir[1] + pt[2] * dir[2];
    Real_v rsumsq               = 1. + radius * radius;
    Real_v a                    = 4. * rdotn;
    Real_v b                    = 2. * (r0sq + 2. * rdotn * rdotn - rsumsq + 2. * dir[2] * dir[2]);
    Real_v c                    = 4. * (r0sq * rdotn - rsumsq * rdotn + 2. * pt[2] * dir[2]);
    Real_v d = r0sq * r0sq - 2. * r0sq * rsumsq + 4. * pt[2] * pt[2] + (1. - radius * radius) * (1. - radius * radius);

    Real_v x[4] = {vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength};
    int nsol    = 0;

    // Skipping a tangent entry can leave t^2 * (t^2 + a*t + b).
    // Solve the deflated quadratic so Newton cannot collapse the next root
    // back onto the skipped surface root.
    if (skipZero && out && Abs(c) <= dirTol && Abs(d) <= tol) {
      Real_v discriminant = a * a - Real_v(4.) * b;
      if (discriminant >= Real_v(0.)) {
        Real_v sqrtDiscriminant = Sqrt(discriminant);
        Real_v candidate[2]     = {Real_v(0.5) * (-a - sqrtDiscriminant), Real_v(0.5) * (-a + sqrtDiscriminant)};
        if (candidate[1] < candidate[0]) {
          Real_v tmp   = candidate[0];
          candidate[0] = candidate[1];
          candidate[1] = tmp;
        }
        for (int i = 0; i < 2; ++i) {
          if (candidate[i] <= tol) continue;
          Vector3D<Real_v> r0 = pt + candidate[i] * dir;
          r0.z()              = 0.;
          r0.Normalize();
          if (torus.dphi() < vecgeom::kTwoPi && !torus.fPhiWedge.ContainsWithBoundary<Real_v>(r0)) continue;
          return candidate[i];
        }
      }
    }

    // special condition
    if (Abs(dir[2]) < Real_v(1E-3) && Abs(pt[2]) < Real_v(0.1) * radius) {
      Real_v r0        = 1. - Sqrt((radius - pt[2]) * (radius + pt[2]));
      Real_v invdirxy2 = 1. / (1 - dir.z() * dir.z());
      Real_v b0        = (pt[0] * dir[0] + pt[1] * dir[1]) * invdirxy2;
      Real_v c0        = (pt[0] * pt[0] + (pt[1] - r0) * (pt[1] + r0)) * invdirxy2;
      Real_v delta     = b0 * b0 - c0;
      if (delta > Real_v(0.)) {
        x[nsol] = -b0 - Sqrt(delta);
        if (x[nsol] > -tol) nsol++;
        x[nsol] = -b0 + Sqrt(delta);
        if (x[nsol] > -tol) nsol++;
      }
      r0    = 1. + Sqrt((radius - pt[2]) * (radius + pt[2]));
      c0    = (pt[0] * pt[0] + (pt[1] - r0) * (pt[1] + r0)) * invdirxy2;
      delta = b0 * b0 - c0;
      if (delta > Real_v(0.)) {
        x[nsol] = -b0 - Sqrt(delta);
        if (x[nsol] > -tol) nsol++;
        x[nsol] = -b0 + Sqrt(delta);
        if (x[nsol] > -tol) nsol++;
      }
      if (nsol) {
        Sort4(x);
      }
    } else { // generic case
      nsol = SolveQuartic(a, b, c, d, x);
    }
    if (!nsol) {
      return vecgeom::kInfLength;
    }

    // look for first positive solution
    Real_v ndotd;
    bool inner = Abs(radius - torus.rmin() / torus.rtor()) < vecgeom::kTolerance;
    for (int i = 0; i < nsol; i++) {
      if (x[i] < Real_v(-100.)) continue;
      if (skipZero && Abs(x[i]) <= tol) continue;

      Vector3D<Real_v> r0   = pt + x[i] * dir;
      Vector3D<Real_v> norm = r0;
      r0.z()                = 0.;
      r0.Normalize();
      // r0 *= torus.rtor();
      norm -= r0;
      // norm = pt
      // for (unsigned int ipt = 0; ipt < 3; ipt++)
      //   norm[ipt] = pt[ipt] + x[i] * dir[ipt] - r0[ipt];
      // ndotd = norm[0] * dir[0] + norm[1] * dir[1] + norm[2] * dir[2];
      ndotd = norm.Dot(dir);
      if (inner ^ out) {
        if (ndotd <= dirTol) continue; // discard this grazing solution
      } else {
        if (ndotd >= -dirTol) continue; // discard this grazing solution
      }

      // The crossing point should be in the phi wedge
      if (torus.dphi() < vecgeom::kTwoPi) {
        if (!torus.fPhiWedge.ContainsWithBoundary<Real_v>(r0)) continue;
      }

      s                  = x[i];
      Real_v preRefinedS = s;
      // refine solution with Newton iterations
      Real_v eps   = vecgeom::kInfLength;
      Real_v delta = s * s * s * s + a * s * s * s + b * s * s + c * s + d;
      Real_v deriv = 4. * s * s * s + 3. * a * s * s + 2. * b * s + c;
      Real_v eps0  = vecgeom::kInfLength;
      if (deriv != Real_v(0.)) eps0 = -delta / deriv;
      int ntry = 0;
      while (Abs(eps) > vecgeom::kTolerance) {
        if (Abs(eps0) > Real_v(200.)) break;
        s += eps0;
        if (Abs(s + eps0) < vecgeom::kTolerance) break;
        delta = s * s * s * s + a * s * s * s + b * s * s + c * s + d;
        deriv = 4. * s * s * s + 3. * a * s * s + 2. * b * s + c;
        if (delta == Real_v(0.) || deriv == Real_v(0.)) break;
        eps = -delta / deriv;
        if (Abs(eps) >= Abs(eps0)) break;
        ntry++;
        // Avoid infinite recursion
        if (ntry > 100) break;
        eps0 = eps;
      }
      // discard this solution
      if (s < -tol) continue;
      if (skipZero && Abs(s) <= tol) {
        // For tangential exits, Newton can collapse the second root back onto
        // the skipped surface root. Keep the pre-refined positive root when it
        // is outside the zero-distance tolerance.
        if (out && preRefinedS > tol) return preRefinedS;
        continue;
      }
      deriv = 4. * s * s * s + 3. * a * s * s + 2. * b * s + c;
      if (inner ^ out) {
        if (deriv <= dirTol) continue;
      } else {
        if (deriv >= -dirTol) continue;
      }
      return Max(Real_v(0.), s);
    }
    return vecgeom::kInfLength;
  }

  /// @brief Compute safety from an interior point to the torus boundary.
  /// @details The radial safety is the distance to the nearest inner or outer
  /// tube circle in the torus cross-section. For phi-cut tori, the result is
  /// tightened with the wedge safety using direct cached wedge access.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] safety Conservative distance to the nearest exit boundary.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &torus,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v rxy = Sqrt(point[0] * point[0] + point[1] * point[1]);
    Real_v rad = Sqrt((rxy - torus.rtor()) * (rxy - torus.rtor()) + point[2] * point[2]);
    safety     = torus.rmax() - rad;
    if (torus.rmin()) {
      safety = Min(rad - torus.rmin(), torus.rmax() - rad);
    }

    bool hasphi = (torus.dphi() < kTwoPi);
    if (hasphi) {
      Real_v safetyPhi = torus.fPhiWedge.SafetyToOut<Real_v>(point);
      safety           = Min(safetyPhi, safety);
    }
  }

  /// @brief Test whether a point is contained by the torus.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] contains True when @p point is not outside.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &torus,
                                                                    Vector3D<Real_v> const &point, bool &contains)
  {
    bool unused  = false;
    bool outside = false;
    TorusImplementation2::GenericKernelForContainsAndInside<Real_v, true, true>(torus, point, unused, outside);
    contains = !outside;
  }

  /// @brief Classify a point as inside, outside, or on the torus surface.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] inside VecGeom inside code.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &torus,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    TorusImplementation2::InsideKernel<Real_v, Inside_t>(torus, point, inside);
  }

  /// @brief Compute distance from an exterior point to the torus boundary.
  /// @details The bounding tube is used as a cheap first stage. After the point
  /// is propagated to that tube, the torus is scaled by `rtor`, phi-plane
  /// candidates are checked against the radial annulus, and torus surface roots
  /// are filtered by `ToBoundary`.
  /// @param torus Torus runtime data.
  /// @param point Query point, expected to be outside the torus.
  /// @param direction Normalized propagation direction.
  /// @param stepMax Maximum propagation step used by the bounding-tube stage.
  /// @param[out] distance Distance to enter, `-1` for inside input.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &torus,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {

    Vector3D<Real_v> localPoint            = point;
    Vector3D<Real_v> const &localDirection = direction;
    Real_v rtor                            = torus.rtor();
    Real_v rmin2Scaled                     = torus.rmin2() / rtor / rtor;
    Real_v rmax2Scaled                     = torus.rmax2() / rtor / rtor;
    Real_v rminScaled                      = torus.rmin() / rtor;
    Real_v rmaxScaled                      = torus.rmax() / rtor;
    bool hasphi                            = torus.dphi() < vecgeom::kTwoPi;
    bool hasrmin                           = torus.rmin() > 0.;

    using Inside_v = vecCore::Index_v<Real_v>;

    ////////First naive implementation
    distance = kInfLength;

    // Check Bounding Cylinder first
    bool inBounds                = false;
    bool done                    = false;
    Inside_v inside              = Inside_v(EInside::kOutside);
    Real_v tubeDistance          = kInfLength;
    bool skipZeroSurfaceBoundary = false;

#ifndef VECGEOM_NO_SPECIALIZATION
    // call the tube functionality -- first of all we check whether we are inside
    // bounding volume
    TubeImplementation<TubeTypes::HollowTube>::Contains(torus.GetBoundingTube().GetStruct(), localPoint, inBounds);

    // only need to do this check if the scalar point is outside.
    if (!inBounds) {
      TubeImplementation<TubeTypes::HollowTube>::DistanceToIn(torus.GetBoundingTube().GetStruct(), localPoint,
                                                              localDirection, stepMax, tubeDistance);
    } else {
      tubeDistance = 0.;
    }
#else
    // call the tube functionality -- first of all we check whether we are inside
    // bounding volume
    TubeImplementation<TubeTypes::UniversalTube>::Contains(torus.GetBoundingTube().GetStruct(), localPoint, inBounds);

    // only need to do this check if the scalar point is outside.
    if (!inBounds) {
      TubeImplementation<TubeTypes::UniversalTube>::DistanceToIn(torus.GetBoundingTube().GetStruct(), localPoint,
                                                                 localDirection, stepMax, tubeDistance);
    } else {
      tubeDistance = 0.;
    }

#endif // VECGEOM_NO_SPECIALIZATION
    if (inBounds) {
      tubeDistance = Real_v(0.);
      // Check points on the wrong side (inside torus)
      TorusImplementation2::InsideKernel<Real_v, Inside_v>(torus, point, inside);
      if (inside == Inside_v(EInside::kInside)) {
        done     = true;
        distance = Real_v(-1.);
      } else if (inside == Inside_v(EInside::kSurface)) {
        bool radialEntry = IsOnRadialSurfaceAndMoving<Real_v, false, false>(torus, point, direction) ||
                           (hasrmin && IsOnRadialSurfaceAndMoving<Real_v, true, false>(torus, point, direction));
        bool radialTangentEntry =
            IsOnRadialSurfaceAndMovingTangentially<Real_v, false, true>(torus, point, direction) ||
            (hasrmin && IsOnRadialSurfaceAndMovingTangentially<Real_v, true, true>(torus, point, direction));
        bool radialExit =
            IsOnRadialSurfaceAndMoving<Real_v, false, true>(torus, point, direction) ||
            IsOnRadialSurfaceAndMovingTangentially<Real_v, false, false>(torus, point, direction) ||
            (hasrmin && (IsOnRadialSurfaceAndMoving<Real_v, true, true>(torus, point, direction) ||
                         IsOnRadialSurfaceAndMovingTangentially<Real_v, true, false>(torus, point, direction)));
        auto const &wedge       = torus.fPhiWedge;
        bool inRadialDisk       = hasphi && InsideRadialCrossSection<Real_v>(torus, point);
        bool phiEntry           = hasphi && inRadialDisk &&
                                  (wedge.template IsPointOnSurfaceAndMovingOut<Real_v, true, false>(point, direction) ||
                                   wedge.template IsPointOnSurfaceAndMovingOut<Real_v, false, false>(point, direction));
        bool phiExit            = hasphi && inRadialDisk &&
                                  (wedge.template IsPointOnSurfaceAndMovingOut<Real_v, true, true>(point, direction) ||
                                   wedge.template IsPointOnSurfaceAndMovingOut<Real_v, false, true>(point, direction));
        skipZeroSurfaceBoundary = radialExit || phiExit;
        if ((radialEntry || radialTangentEntry || phiEntry) && !skipZeroSurfaceBoundary) {
          distance = Real_v(0.);
          return;
        }
      }
    } else {
      done = tubeDistance == kInfLength;
    }

    if (done) return;

    // Propagate the point to the bounding tube, as this will reduce the
    // coefficients of the quartic and improve precision of the solutions
    localPoint += tubeDistance * localDirection;
    localPoint /= rtor;
    bool skipZeroPhiBoundary = skipZeroSurfaceBoundary;
    if (hasphi) {
      Real_v d1, d2;

      auto const &wedge = torus.fPhiWedge;
      skipZeroPhiBoundary |=
          wedge.template IsPointOnSurfaceAndMovingOut<Real_v, true, true>(localPoint, localDirection) ||
          wedge.template IsPointOnSurfaceAndMovingOut<Real_v, false, true>(localPoint, localDirection);
      // checking distance to phi wedges
      // NOTE: if the tube told me its hitting surface, this would be unnecessary
      wedge.DistanceToIn<Real_v>(localPoint, localDirection, d1, d2);

      // check phi intersections if bounding tube intersection is due to phi in which case we are done
      if (d1 != kInfLength && !(skipZeroPhiBoundary && Abs(d1) <= kTolerance)) {
        Real_v daxis = DistSqrToTorusR(localPoint, localDirection, d1);
        if (daxis >= rmin2Scaled && daxis < rmax2Scaled && d1 > -kTolerance) {
          distance = d1;
        }
      }

      if (d2 != kInfLength && !(skipZeroPhiBoundary && Abs(d2) <= kTolerance)) {
        Real_v daxis = DistSqrToTorusR(localPoint, localDirection, d2);
        if (daxis >= rmin2Scaled && daxis < rmax2Scaled && d2 > -kTolerance) {
          distance = Min(distance, d2);
        }
      }
    }

    Real_v dd = ToBoundary<Real_v, false>(torus, localPoint, localDirection, rmaxScaled, false, skipZeroPhiBoundary);

    // in case of a phi opening we also need to check the Rmin surface
    if (torus.rmin() > 0.) {
      Real_v ddrmin =
          ToBoundary<Real_v, true>(torus, localPoint, localDirection, rminScaled, false, skipZeroPhiBoundary);
      dd = Min(dd, ddrmin);
    }
    distance = Min(distance, dd);
    distance *= rtor;
    distance += tubeDistance;
    // This has to be added because distance can become > kInfLength due to
    // missing early returns in CUDA. This makes comparisons to kInfLength fail.
    if (Abs(distance) > kInfLength) distance = kInfLength;

    return;
  }

  /// @brief Compute safety from an exterior point to the torus boundary.
  /// @details Radial safety is based on the torus cross-section. For phi-cut
  /// tori, wedge safety is combined only when the transverse radius is nonzero,
  /// avoiding undefined phi direction at the z axis.
  /// @param torus Torus runtime data.
  /// @param point Query point.
  /// @param[out] safety Conservative distance to the nearest entry boundary.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &torus,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {

    Vector3D<Real_v> localPoint = point;

    // implementation taken from TGeoTorus
    Real_v rxy = Sqrt(localPoint[0] * localPoint[0] + localPoint[1] * localPoint[1]);
    Real_v rad = Sqrt((rxy - torus.rtor()) * (rxy - torus.rtor()) + localPoint[2] * localPoint[2]);
    safety     = rad - torus.rmax();
    if (torus.rmin()) {
      safety = Max(torus.rmin() - rad, rad - torus.rmax());
    }

    bool hasphi = (torus.dphi() < kTwoPi);
    if (hasphi && rxy != Real_v(0.)) {
      Real_v safetyPhi = torus.fPhiWedge.SafetyToIn<Real_v>(localPoint);
      safety           = Max(safetyPhi, safety);
    }
  }

}; // end struct
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION2_H_
