// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file implements the algorithms for EllipticalTube
/// @file volumes/kernel/EllipticalTubeImplementation.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPTICALTUBEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPTICALTUBEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipticalTubeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipticalTubeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipticalTubeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipticalTube;
template <typename T>
struct EllipticalTubeStruct;
class UnplacedEllipticalTube;

struct EllipticalTubeImplementation {

  using PlacedShape_t    = PlacedEllipticalTube;
  using UnplacedStruct_t = EllipticalTubeStruct<Precision>;
  using UnplacedVolume_t = UnplacedEllipticalTube;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &ellipticaltube,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(ellipticaltube, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &ellipticaltube,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(ellipticaltube, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &ellipticaltube, Vector3D<Real_v> const &point, Bool_v &completelyinside,
      Bool_v &completelyoutside)
  {
    /* TODO : Logic to check where the point is inside or not.
    **
    ** if ForInside is false then it will only check if the point is outside,
    ** and is used by Contains function
    **
    ** if ForInside is true then it will check whether the point is inside or outside,
    ** and if neither inside nor outside then it is on the surface.
    ** and is used by Inside function
    */
    Real_v x      = point.x() * ellipticaltube.fSx;
    Real_v y      = point.y() * ellipticaltube.fSy;
    Real_v distR  = ellipticaltube.fQ1 * (x * x + y * y) - ellipticaltube.fQ2;
    Real_v distZ  = vecCore::math::Abs(point.z()) - ellipticaltube.fDz;
    Real_v safety = vecCore::math::Max(distR, distZ);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &ellipticaltube,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from outside point to the EllipticalTube surface */
    using Bool_v = vecCore::Mask_v<Real_v>;
    distance     = kInfLength;
    Real_v offset(0.);
    Vector3D<Real_v> pcur(point);

    // Move point closer, if required
    Real_v Rfar2(1024. * ellipticaltube.fRsph * ellipticaltube.fRsph); // 1024 = 32 * 32
    vecCore__MaskedAssignFunc(pcur, ((pcur.Mag2() > Rfar2) && (direction.Dot(point) < Real_v(0.))),
                              pcur + (offset = pcur.Mag() - Real_v(2.) * ellipticaltube.fRsph) * direction);

    // Scale elliptical tube to cylinder
    Real_v px = pcur.x() * ellipticaltube.fSx;
    Real_v py = pcur.y() * ellipticaltube.fSy;
    Real_v pz = pcur.z();
    Real_v vx = direction.x() * ellipticaltube.fSx;
    Real_v vy = direction.y() * ellipticaltube.fSy;
    Real_v vz = direction.z();

    // Find intersection with Z planes
    Real_v invz  = Real_v(-1.) / NonZero(vz);
    Real_v dz    = vecCore::math::CopySign(Real_v(ellipticaltube.fDz), invz);
    Real_v tzmin = (pz - dz) * invz;
    Real_v tzmax = (pz + dz) * invz;

    // Find intersection with lateral surface, solve equation: A t^2 + 2B t + C = 0
    Real_v rr = px * px + py * py;
    Real_v A  = vx * vx + vy * vy;
    Real_v B  = px * vx + py * vy;
    Real_v C  = rr - ellipticaltube.fR * ellipticaltube.fR;
    Real_v D  = B * B - A * C;

    // Check if point leaving shape
    Real_v distZ       = vecCore::math::Abs(pz) - ellipticaltube.fDz;
    Real_v distR       = ellipticaltube.fQ1 * rr - ellipticaltube.fQ2;
    Bool_v parallelToZ = (A < kEpsilon || vecCore::math::Abs(vz) >= Real_v(1.));
    Bool_v leaving     = (distZ >= -kHalfTolerance && pz * vz >= Real_v(0.)) ||
                     (distR >= -kHalfTolerance && (B >= Real_v(0.) || parallelToZ));

    // Two special cases where D <= 0:
    //   1) trajectory parallel to Z axis (A = 0, B = 0, C - any, D = 0)
    //   2) touch (D = 0) or no intersection (D < 0) with lateral surface
    vecCore__MaskedAssignFunc(distance, !leaving && parallelToZ, tzmin + offset);   // 1)
    Bool_v done = (leaving || parallelToZ || D <= A * A * ellipticaltube.fScratch); // 2)

    // if (D <= A * A * ellipticaltube.fScratch) std::cerr << "=== SCRATCH D = " << D << std::endl;

    // Find roots of the quadratic
    Real_v tmp(0.), t1(0.), t2(0.);
    vecCore__MaskedAssignFunc(tmp, !done, -B - vecCore::math::CopySign(vecCore::math::Sqrt(D), B));
    vecCore__MaskedAssignFunc(t1, !done, tmp / A);
    vecCore__MaskedAssignFunc(t2, !done, C / tmp);
    Real_v trmin = vecCore::math::Min(t1, t2);
    Real_v trmax = vecCore::math::Max(t1, t2);

    // Set distance
    // No special check for inside points, for inside points distance will be negative
    Real_v tin  = vecCore::math::Max(tzmin, trmin);
    Real_v tout = vecCore::math::Min(tzmax, trmax);
    vecCore__MaskedAssignFunc(distance, !done && (tout - tin) >= kHalfTolerance, tin + offset);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &ellipticaltube,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from inside point to the EllipticalTube surface */
    using Bool_v = vecCore::Mask_v<Real_v>;

    // Scale elliptical tube to cylinder
    Real_v px = point.x() * ellipticaltube.fSx;
    Real_v py = point.y() * ellipticaltube.fSy;
    Real_v pz = point.z();
    Real_v vx = direction.x() * ellipticaltube.fSx;
    Real_v vy = direction.y() * ellipticaltube.fSy;
    Real_v vz = direction.z();

    // Check if point is outside ("wrong side")
    Real_v rr      = px * px + py * py;
    Real_v distR   = ellipticaltube.fQ1 * rr - ellipticaltube.fQ2;
    Real_v distZ   = vecCore::math::Abs(pz) - ellipticaltube.fDz;
    Bool_v outside = vecCore::math::Max(distR, distZ) > kHalfTolerance;
    distance       = Real_v(0.);
    vecCore__MaskedAssignFunc(distance, outside, Real_v(-1.));

    // Find intersection with Z planes
    Real_v tzmax = kMaximum;
    vecCore__MaskedAssignFunc(tzmax, vz != Real_v(0.),
                              (vecCore::math::CopySign(Real_v(ellipticaltube.fDz), vz) - pz) / vz);

    // Find intersection with lateral surface, solve equation: A t^2 + 2B t + C = 0
    Real_v A = vx * vx + vy * vy;
    Real_v B = px * vx + py * vy;
    Real_v C = rr - ellipticaltube.fR * ellipticaltube.fR;
    Real_v D = B * B - A * C;

    // Two cases where D <= 0:
    //   1) trajectory parallel to Z axis (A = 0, B = 0, C - any, D = 0)
    //   2) touch (D = 0) or no intersection (D < 0) with lateral surface
    Bool_v parallelToZ = (A < kEpsilon || vecCore::math::Abs(vz) >= Real_v(1.));
    vecCore__MaskedAssignFunc(distance, (!outside && parallelToZ), tzmax); // 1)
    Bool_v done = (outside || parallelToZ || D <= Real_v(0.));             // 2)
    // Bool_v done = (outside || parallelToZ || D < A * A * ellipticaltube.fScratch); // alternative 2)

    // Set distance
    vecCore__MaskedAssignFunc(distance, !done && B >= Real_v(0.),
                              vecCore::math::Min(tzmax, -C / (vecCore::math::Sqrt(D) + B)));
    vecCore__MaskedAssignFunc(distance, !done && B < Real_v(0.),
                              vecCore::math::Min(tzmax, (vecCore::math::Sqrt(D) - B) / A));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &ellipticaltube,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the EllipticalTube surface */
    Real_v x     = point.x() * ellipticaltube.fSx;
    Real_v y     = point.y() * ellipticaltube.fSy;
    Real_v distR = vecCore::math::Sqrt(x * x + y * y) - ellipticaltube.fR;
    Real_v distZ = vecCore::math::Abs(point.z()) - ellipticaltube.fDz;

    safety = vecCore::math::Max(distR, distZ);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &ellipticaltube,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the EllipticalTube surface */
    Real_v x     = point.x() * ellipticaltube.fSx;
    Real_v y     = point.y() * ellipticaltube.fSy;
    Real_v distR = ellipticaltube.fR - vecCore::math::Sqrt(x * x + y * y);
    Real_v distZ = ellipticaltube.fDz - vecCore::math::Abs(point.z());

    safety = vecCore::math::Min(distR, distZ);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &ellipticaltube, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    //   In case if the point is further than kHalfTolerance from the surface, set valid=false
    //   Must return a valid vector (even if the point is not on the surface)
    //
    //   On an edge provide an average normal of the corresponding base and lateral surface
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v x     = point.x() * ellipticaltube.fSx;
    Real_v y     = point.y() * ellipticaltube.fSy;
    Real_v distR = ellipticaltube.fQ1 * (x * x + y * y) - ellipticaltube.fQ2;
    vecCore__MaskedAssignFunc(
        normal, vecCore::math::Abs(distR) <= kHalfTolerance,
        Vector3D<Real_v>(point.x() * ellipticaltube.fDDy, point.y() * ellipticaltube.fDDx, 0.).Unit());

    Real_v distZ = vecCore::math::Abs(point.z()) - ellipticaltube.fDz;
    vecCore__MaskedAssignFunc(normal[2], vecCore::math::Abs(distZ) <= kHalfTolerance, vecCore::math::Sign(point[2]));
    vecCore__MaskedAssignFunc(normal, normal.Mag2() > 1., normal.Unit());

    vecCore::Mask_v<Real_v> done = normal.Mag2() > Real_v(0.);
    if (vecCore::MaskFull(done)) return normal;

    // Point is not on the surface - normally, this should never be
    // Return normal to the nearest surface
    vecCore__MaskedAssignFunc(valid, !done, false);
    vecCore__MaskedAssignFunc(normal[2], !done, vecCore::math::Sign(point[2]));
    vecCore__MaskedAssignFunc(distR, !done, vecCore::math::Sqrt(x * x + y * y) - ellipticaltube.fR);
    vecCore__MaskedAssignFunc(
        normal, !done && distR > distZ && (x * x + y * y) > Real_v(0.),
        Vector3D<Real_v>(point.x() * ellipticaltube.fDDy, point.y() * ellipticaltube.fDDx, 0.).Unit());
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPTICALTUBEIMPLEMENTATION_H_
