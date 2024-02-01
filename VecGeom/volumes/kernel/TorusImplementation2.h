/// @file TorusImplementation2.h

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

template <typename T, unsigned int i, unsigned int j>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CmpAndSwap(T *array)
{
  if (vecCore::MaskFull(array[i] > array[j])) {
    T c      = array[j];
    array[j] = array[i];
    array[i] = c;
  }
}

// a special function to sort a 4 element array
// sorting is done inplace and in increasing order
// implementation comes from a sorting network
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Sort4(T *array)
{
  CmpAndSwap<T, 0, 2>(array);
  CmpAndSwap<T, 1, 3>(array);
  CmpAndSwap<T, 0, 1>(array);
  CmpAndSwap<T, 2, 3>(array);
  CmpAndSwap<T, 1, 2>(array);
}

// solve quartic taken from ROOT/TGeo and adapted
//_____________________________________________________________________________

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
  if (Abs(f) < 1E-6) {
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

struct TorusImplementation2 {
  using PlacedShape_t    = PlacedTorus2;
  using UnplacedStruct_t = TorusStruct2<Precision>;
  using UnplacedVolume_t = UnplacedTorus2;

  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static Real_v DistSqrToTorusR(UnplacedStruct_t const &torus, Vector3D<Real_v> const &point,
                                                        Vector3D<Real_v> const &dir, Real_v dist)
  {
    Vector3D<Real_v> p = point + dir * dist;
    Real_v rxy         = p.Perp();
    return (rxy - torus.rtor()) * (rxy - torus.rtor()) + p.z() * p.z();
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &torus,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const & /*stepMax*/, Real_v &distance)
  {
    using Inside_v = vecCore::Index_v<Real_v>;
    using Bool_v   = vecCore::Mask_v<Real_v>;

    bool hasphi  = (torus.dphi() < kTwoPi);
    bool hasrmin = (torus.rmin() > 0);

    //=== First, for points outside --> return infinity
    Bool_v done = Bool_v(false);
    distance    = kInfLength;

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
    vecCore__MaskedAssignFunc(distance, done, Real_v(-1.));
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    //=== Use InsideKernel() for a quick check, and if outside --> return -1
    // Bool_t inside=false, outside=false;
    // GenericKernelForContainsAndInside<Backend,true,true>(torus, point, inside, outside);
    // MaskedAssign( inside, -1.0, &distance );
    // done |= inside;
    Inside_v locus;
    TorusImplementation2::InsideKernel<Real_v, Inside_v>(torus, point, locus);
    vecCore__MaskedAssignFunc(distance, locus == EInside::kOutside, Real_v(-1.));
    done |= locus == EInside::kOutside;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    Real_v dout = ToBoundary<Real_v, false>(torus, point, dir, torus.rmax(), true);
    // ToBoundary<Backend, false, true>(torus, point, dir, torus.rmax());
    Real_v din(kInfLength);
    if (hasrmin) {
      din = ToBoundary<Real_v, true>(torus, point, dir, torus.rmin(), true);
      // ToBoundary<Backend, true, true>(torus, point, dir, torus.rmin());
    }
    distance = Min(dout, din);
    // std::cerr << "dout, din: " << dout << ", " << din << '\n';
    // std::cerr << "distance = Min(dout, din): " << distance << '\n';

    if (hasphi) {
      Real_v distPhi1;
      Real_v distPhi2;
      // torus.GetWedge().DistanceToOut<Backend>(point, dir, distPhi1, distPhi2);
      torus.GetWedge().DistanceToOut<Real_v>(point, dir, distPhi1, distPhi2);
      Bool_v smallerphi = distPhi1 < distance;
      if (!vecCore::MaskEmpty(smallerphi)) {
        Vector3D<Real_v> intersectionPoint = point + dir * distPhi1;
        Bool_v insideDisk;
        // UnplacedContainsDisk<Backend>(torus, intersectionPoint, insideDisk);
        UnplacedContainsDisk<Real_v, Bool_v>(torus, intersectionPoint, insideDisk);

        if (!vecCore::MaskEmpty(insideDisk)) // Inside Disk
        {
          Real_v diri = intersectionPoint.x() * torus.GetWedge().GetAlong1().x() +
                        intersectionPoint.y() * torus.GetWedge().GetAlong1().y();
          Bool_v rightside = (diri >= 0);

          vecCore__MaskedAssignFunc(distance, rightside && smallerphi && insideDisk, distPhi1);
        }
      }
      smallerphi = distPhi2 < distance;
      if (!vecCore::MaskEmpty(smallerphi)) {

        Vector3D<Real_v> intersectionPoint = point + dir * distPhi2;
        Bool_v insideDisk;
        // UnplacedContainsDisk<Backend>(torus, intersectionPoint, insideDisk);
        UnplacedContainsDisk<Real_v, Bool_v>(torus, intersectionPoint, insideDisk);
        if (!vecCore::MaskEmpty(insideDisk)) // Inside Disk
        {
          Real_v diri2 = intersectionPoint.x() * torus.GetWedge().GetAlong2().x() +
                         intersectionPoint.y() * torus.GetWedge().GetAlong2().y();
          Bool_v rightside = (diri2 >= Real_v(0));
          vecCore__MaskedAssignFunc(distance, rightside && (distPhi2 < distance) && smallerphi && insideDisk, distPhi2);
        }
      }
    }

    vecCore__MaskedAssignFunc(distance, done || distance >= kInfLength, Real_v(-1.));
  }

  template <typename Real_v, typename Bool_v, bool notForDisk>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ContainsKernel(UnplacedStruct_t const &torus,
                                                                          Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused;
    Bool_v outside;
    TorusImplementation2::GenericKernelForContainsAndInside<Real_v, false, notForDisk>(torus, point, unused, outside);
    inside = !outside;
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContainsDisk(UnplacedStruct_t const &torus,
                                                                                Vector3D<Real_v> const &point,
                                                                                Bool_v &inside)
  {
    ContainsKernel<Real_v, Bool_v, false>(torus, point, inside);
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void InsideKernel(UnplacedStruct_t const &torus,
                                                                        Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    //
    Bool_v completelyinside, completelyoutside;
    TorusImplementation2::GenericKernelForContainsAndInside<Real_v, true, true>(torus, point, completelyinside,
                                                                                completelyoutside);
    inside = Inside_t(EInside::kSurface);
    vecCore::MaskedAssign(inside, completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, bool ForInside, bool notForDisk>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &torus, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &completelyinside,
      typename vecCore::Mask_v<Real_v> &completelyoutside)

  {
    // using vecgeom::GenericKernels;
    // here we are explicitely unrolling the loop since  a for statement will likely be a penality
    // check if second call to Abs is compiled away
    // and it can anyway not be vectorized
    /* rmax */
    using Bool_v                = vecCore::Mask_v<Real_v>;
    VECGEOM_CONST Precision tol = 100. * vecgeom::kTolerance;

    Real_v rxy   = Sqrt(point[0] * point[0] + point[1] * point[1]);
    Real_v radsq = (rxy - torus.rtor()) * (rxy - torus.rtor()) + point[2] * point[2];

    if (ForInside) {
      completelyoutside = radsq > (tol * torus.rmax() + torus.rmax2()); // rmax
      completelyinside  = radsq < (-tol * torus.rmax() + torus.rmax2());
    } else {
      completelyoutside = radsq > torus.rmax2();
    }

    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }
    /* rmin */
    if (ForInside) {
      completelyoutside |= radsq < (-tol * torus.rmin() + torus.rmin2()); // rmin
      completelyinside &= radsq > (tol * torus.rmin() + torus.rmin2());
    } else {
      completelyoutside |= radsq < torus.rmin2();
    }

    // NOT YET NEEDED WHEN NOT PHI TREATMENT
    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    /* phi */
    if ((torus.dphi() < kTwoPi) && (notForDisk)) {
      Bool_v completelyoutsidephi;
      Bool_v completelyinsidephi;
      torus.GetWedge().GenericKernelForContainsAndInside<Real_v, ForInside>(point, completelyinsidephi,
                                                                            completelyoutsidephi);

      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  template <typename Real_v, bool ForRmin>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v ToBoundary(UnplacedStruct_t const &torus,
                                                                        Vector3D<Real_v> const &pt,
                                                                        Vector3D<Real_v> const &dir, Real_v radius,
                                                                        bool out)
  {
    // to be taken from ROOT
    // Returns distance to the surface or the torus from a point, along
    // a direction. Point is close enough to the boundary so that the distance
    // to the torus is decreasing while moving along the given direction.

    // Compute coeficients of the quartic
    Real_v s                 = vecgeom::kInfLength;
    VECGEOM_CONST Real_v tol = 100. * vecgeom::kTolerance;
    Real_v r0sq              = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
    Real_v rdotn             = pt[0] * dir[0] + pt[1] * dir[1] + pt[2] * dir[2];
    Real_v rsumsq            = torus.rtor2() + radius * radius;
    Real_v a                 = 4. * rdotn;
    Real_v b                 = 2. * (r0sq + 2. * rdotn * rdotn - rsumsq + 2. * torus.rtor2() * dir[2] * dir[2]);
    Real_v c                 = 4. * (r0sq * rdotn - rsumsq * rdotn + 2. * torus.rtor2() * pt[2] * dir[2]);
    Real_v d                 = r0sq * r0sq - 2. * r0sq * rsumsq + 4. * torus.rtor2() * pt[2] * pt[2] +
               (torus.rtor2() - radius * radius) * (torus.rtor2() - radius * radius);

    Real_v x[4] = {vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength};
    int nsol    = 0;

    // special condition
    if (vecCore::MaskFull(Abs(dir[2]) < 1E-3 && Abs(pt[2]) < 0.1 * radius)) {
      Real_v r0        = torus.rtor() - Sqrt((radius - pt[2]) * (radius + pt[2]));
      Real_v invdirxy2 = 1. / (1 - dir.z() * dir.z());
      Real_v b0        = (pt[0] * dir[0] + pt[1] * dir[1]) * invdirxy2;
      Real_v c0        = (pt[0] * pt[0] + (pt[1] - r0) * (pt[1] + r0)) * invdirxy2;
      Real_v delta     = b0 * b0 - c0;
      if (vecCore::MaskFull(delta > 0)) {
        x[nsol] = -b0 - Sqrt(delta);
        if (vecCore::MaskFull(x[nsol] > -tol)) nsol++;
        x[nsol] = -b0 + Sqrt(delta);
        if (vecCore::MaskFull(x[nsol] > -tol)) nsol++;
      }
      r0    = torus.rtor() + Sqrt((radius - pt[2]) * (radius + pt[2]));
      c0    = (pt[0] * pt[0] + (pt[1] - r0) * (pt[1] + r0)) * invdirxy2;
      delta = b0 * b0 - c0;
      if (vecCore::MaskFull(delta > 0)) {
        x[nsol] = -b0 - Sqrt(delta);
        if (vecCore::MaskFull(x[nsol] > -tol)) nsol++;
        x[nsol] = -b0 + Sqrt(delta);
        if (vecCore::MaskFull(x[nsol] > -tol)) nsol++;
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
    bool inner = vecCore::MaskFull(Abs(radius - torus.rmin()) < vecgeom::kTolerance);
    for (int i = 0; i < nsol; i++) {
      if (vecCore::MaskFull(x[i] < -10)) continue;

      Vector3D<Real_v> r0   = pt + x[i] * dir;
      Vector3D<Real_v> norm = r0;
      r0.z()                = 0.;
      r0.Normalize();
      r0 *= torus.rtor();
      norm -= r0;
      // norm = pt
      // for (unsigned int ipt = 0; ipt < 3; ipt++)
      //   norm[ipt] = pt[ipt] + x[i] * dir[ipt] - r0[ipt];
      // ndotd = norm[0] * dir[0] + norm[1] * dir[1] + norm[2] * dir[2];
      ndotd = norm.Dot(dir);
      if (inner ^ out) {
        if (vecCore::MaskFull(ndotd < 0)) continue; // discard this solution
      } else {
        if (vecCore::MaskFull(ndotd > 0)) continue; // discard this solution
      }

      // The crossing point should be in the phi wedge
      if (torus.dphi() < vecgeom::kTwoPi) {
        if (!vecCore::MaskFull(torus.GetWedge().ContainsWithBoundary<Real_v>(r0))) continue;
      }

      s = x[i];
      // refine solution with Newton iterations
      Real_v eps   = vecgeom::kInfLength;
      Real_v delta = s * s * s * s + a * s * s * s + b * s * s + c * s + d;
      Real_v eps0  = -delta / (4. * s * s * s + 3. * a * s * s + 2. * b * s + c);
      int ntry     = 0;
      while (vecCore::MaskFull(Abs(eps) > vecgeom::kTolerance)) {
        if (vecCore::MaskFull(Abs(eps0) > 100)) break;
        s += eps0;
        if (vecCore::MaskFull(Abs(s + eps0) < vecgeom::kTolerance)) break;
        delta = s * s * s * s + a * s * s * s + b * s * s + c * s + d;
        eps   = -delta / (4. * s * s * s + 3. * a * s * s + 2. * b * s + c);
        if (vecCore::MaskFull(Abs(eps) >= Abs(eps0))) break;
        ntry++;
        // Avoid infinite recursion
        if (ntry > 100) break;
        eps0 = eps;
      }
      // discard this solution
      if (vecCore::MaskFull(s < -tol)) continue;
      return Max(Real_v(0.), s);
    }
    return vecgeom::kInfLength;
  }

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

    // TODO: extend implementation for phi sector case
    bool hasphi = (torus.dphi() < kTwoPi);
    if (hasphi) {
      Real_v safetyPhi = torus.GetWedge().SafetyToOut<Real_v>(point);
      safety           = Min(safetyPhi, safety);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &torus,
                                                                    Vector3D<Real_v> const &point,
                                                                    typename vecCore::Mask_v<Real_v> &contains)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v unused, outside;
    TorusImplementation2::GenericKernelForContainsAndInside<Real_v, true, true>(torus, point, unused, outside);
    contains = !outside;
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &torus,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    TorusImplementation2::InsideKernel<Real_v, Inside_t>(torus, point, inside);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &torus,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {

    Vector3D<Real_v> localPoint     = point;
    Vector3D<Real_v> localDirection = direction;

    using Bool_v   = vecCore::Mask_v<Real_v>;
    using Inside_v = vecCore::Index_v<Real_v>;

    ////////First naive implementation
    distance = kInfLength;

    // Check Bounding Cylinder first
    Bool_v inBounds;
    Bool_v done         = Bool_v(false);
    Inside_v inside     = Inside_v(EInside::kOutside);
    Real_v tubeDistance = kInfLength;

#ifndef VECGEOM_NO_SPECIALIZATION
    // call the tube functionality -- first of all we check whether we are inside
    // bounding volume
    TubeImplementation<TubeTypes::HollowTube>::Contains(torus.GetBoundingTube().GetStruct(), localPoint, inBounds);

    // only need to do this check if all particles (in vector) are outside ( otherwise useless )
    TubeImplementation<TubeTypes::HollowTube>::DistanceToIn(torus.GetBoundingTube().GetStruct(), localPoint,
                                                            localDirection, stepMax, tubeDistance);
#else
    // call the tube functionality -- first of all we check whether we are inside
    // bounding volume
    TubeImplementation<TubeTypes::UniversalTube>::Contains(torus.GetBoundingTube().GetStruct(), localPoint, inBounds);

    // only need to do this check if all particles (in vector) are outside ( otherwise useless )
    // vecCore::Mask_v<Real_v> notInBounds { !inBounds };
    if (!inBounds) {
      TubeImplementation<TubeTypes::UniversalTube>::DistanceToIn(torus.GetBoundingTube().GetStruct(), localPoint,
                                                                 localDirection, stepMax, tubeDistance);
    } else {
      tubeDistance = 0.;
    }

#endif // VECGEOM_NO_SPECIALIZATION
    if (inBounds) {
      // Check points on the wrong side (inside torus)
      TorusImplementation2::InsideKernel<Real_v, Inside_v>(torus, point, inside);
      if (vecCore::MaskFull(inside == Inside_v(EInside::kInside))) {
        done     = Bool_v(true);
        distance = Real_v(-1.);
      }
    } else {
      done = Bool_v(vecCore::MaskFull(tubeDistance == kInfLength));
    }

    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(done)) {
        return;
      }
    }

    // Propagate the point to the bounding tube, as this will reduce the
    // coefficients of the quartic and improve precision of the solutions
    localPoint += tubeDistance * localDirection;
    Bool_v hasphi = Bool_v(torus.dphi() < vecgeom::kTwoPi);
    if (vecCore::MaskFull(hasphi)) {
      Real_v d1, d2;

      auto wedge = torus.GetWedge();
      // checking distance to phi wedges
      // NOTE: if the tube told me its hitting surface, this would be unnessecary
      wedge.DistanceToIn<Real_v>(localPoint, localDirection, d1, d2);

      // check phi intersections if bounding tube intersection is due to phi in which case we are done
      if (vecCore::MaskFull(d1 != kInfLength)) {
        Real_v daxis = DistSqrToTorusR(torus, localPoint, localDirection, d1);
        if (vecCore::MaskFull(daxis >= torus.rmin2() && daxis < torus.rmax2() && d1 > -kTolerance)) {
          distance = d1;
        }
      }

      if (vecCore::MaskFull(d2 != kInfLength)) {
        Real_v daxis = DistSqrToTorusR(torus, localPoint, localDirection, d2);
        if (vecCore::MaskFull(daxis >= torus.rmin2() && daxis < torus.rmax2() && d2 > -kTolerance)) {
          distance = Min(distance, d2);
        }
      }
    }

    Real_v dd = ToBoundary<Real_v, false>(torus, localPoint, localDirection, torus.rmax(), false);

    // in case of a phi opening we also need to check the Rmin surface
    if (torus.rmin() > 0.) {
      Real_v ddrmin = ToBoundary<Real_v, true>(torus, localPoint, localDirection, torus.rmin(), false);
      dd            = Min(dd, ddrmin);
    }
    distance = Min(distance, dd);
    distance += tubeDistance;
    // This has to be added because distance can become > kInfLength due to
    // missing early returns in CUDA. This makes comparisons to kInfLength fail.
    if (vecCore::MaskFull(Abs(distance) > kInfLength)) distance = kInfLength;

    return;
  }

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
    if (hasphi && vecCore::MaskFull(rxy != 0.)) {
      Real_v safetyPhi = torus.GetWedge().SafetyToIn<Real_v>(localPoint);
      safety           = Max(safetyPhi, safety);
    }
  }

}; // end struct
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION2_H_
