// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file implements the algorithms for Ellipsoid
/// @file volumes/kernel/EllipsoidImplementation.h
/// @author Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPSOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPSOIDIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipsoidStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>
#include <iomanip>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipsoidImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipsoidImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipsoid;
template <typename T>
struct EllipsoidStruct;
class UnplacedEllipsoid;

struct EllipsoidImplementation {

  using PlacedShape_t    = PlacedEllipsoid;
  using UnplacedStruct_t = EllipsoidStruct<Precision>;
  using UnplacedVolume_t = UnplacedEllipsoid;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &ellipsoid,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(ellipsoid, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &ellipsoid,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(ellipsoid, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &ellipsoid, Vector3D<Real_v> const &point, Bool_v &completelyinside,
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
    Real_v x      = point.x() * ellipsoid.fSx;
    Real_v y      = point.y() * ellipsoid.fSy;
    Real_v z      = point.z() * ellipsoid.fSz;
    Real_v distZ  = vecCore::math::Abs(z - ellipsoid.fScZMidCut) - ellipsoid.fScZDimCut;
    Real_v distR  = ellipsoid.fQ1 * (x * x + y * y + z * z) - ellipsoid.fQ2;
    Real_v safety = vecCore::math::Max(distZ, distR);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &ellipsoid,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from outside point to the Ellipsoid surface */
    using Bool_v = vecCore::Mask_v<Real_v>;
    distance     = kInfLength;
    Real_v offset(0.);
    Vector3D<Real_v> pcur(point);

    // Move point closer, if required
    Real_v Rfar2(1024. * ellipsoid.fRsph * ellipsoid.fRsph); // 1024 = 32 * 32
    vecCore__MaskedAssignFunc(pcur, ((pcur.Mag2() > Rfar2) && (direction.Dot(point) < Real_v(0.))),
                              pcur + (offset = pcur.Mag() - Real_v(2.) * ellipsoid.fRsph) * direction);

    // Scale ellipsoid to sphere
    Real_v px = pcur.x() * ellipsoid.fSx;
    Real_v py = pcur.y() * ellipsoid.fSy;
    Real_v pz = pcur.z() * ellipsoid.fSz;
    Real_v vx = direction.x() * ellipsoid.fSx;
    Real_v vy = direction.y() * ellipsoid.fSy;
    Real_v vz = direction.z() * ellipsoid.fSz;

    // Check if point is leaving the solid
    Real_v pzcut = pz - ellipsoid.fScZMidCut;
    Real_v dzcut = Real_v(ellipsoid.fScZDimCut);
    Real_v distZ = vecCore::math::Abs(pzcut) - dzcut;

    Real_v rr    = px * px + py * py + pz * pz;
    Real_v vv    = vx * vx + vy * vy + vz * vz;
    Real_v pv    = px * vx + py * vy + pz * vz;
    Real_v distR = ellipsoid.fQ1 * rr - ellipsoid.fQ2;
    Bool_v leaving =
        (distZ >= -kHalfTolerance && pzcut * vz >= Real_v(0.)) || (distR >= -kHalfTolerance && pv >= Real_v(0.));

    // Find intersection with Z planes
    Real_v invz  = Real_v(-1.) / NonZero(vz);
    Real_v dz    = vecCore::math::CopySign(dzcut, invz);
    Real_v tzmin = (pzcut - dz) * invz;
    Real_v tzmax = (pzcut + dz) * invz;

    // Find intersection with sphere
    Real_v A     = vv;
    Real_v B     = pv;
    Real_v C     = (rr - ellipsoid.fR * ellipsoid.fR);
    Real_v D     = B * B - A * C;
    Real_v sqrtD = vecCore::math::Sqrt(vecCore::math::Abs(D));
    Real_v trmin = (-B - sqrtD) / A;
    Real_v trmax = (-B + sqrtD) / A;

    // Set preliminary distances to in/out
    Real_v tmin = vecCore::math::Max(tzmin, trmin);
    Real_v tmax = vecCore::math::Min(tzmax, trmax);

    // Check if no intersection
    Real_v EPS  = Real_v(2.) * rr * vv * kEpsilon;
    Bool_v done = leaving || (D <= EPS) || ((tmax - tmin) <= kHalfTolerance);

    // Set distance
    vecCore__MaskedAssignFunc(distance, !done, tmin + offset);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &ellipsoid,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from inside point to the Ellipsoid surface */
    using Bool_v = vecCore::Mask_v<Real_v>;

    // Scale ellipsoid to sphere
    Real_v px = point.x() * ellipsoid.fSx;
    Real_v py = point.y() * ellipsoid.fSy;
    Real_v pz = point.z() * ellipsoid.fSz;
    Real_v vx = direction.x() * ellipsoid.fSx;
    Real_v vy = direction.y() * ellipsoid.fSy;
    Real_v vz = direction.z() * ellipsoid.fSz;

    // Check if point is outside ("wrong side")
    Real_v pzcut = pz - ellipsoid.fScZMidCut;
    Real_v dzcut = Real_v(ellipsoid.fScZDimCut);
    Real_v distZ = vecCore::math::Abs(pzcut) - dzcut;

    Real_v rr      = px * px + py * py + pz * pz;
    Real_v vv      = vx * vx + vy * vy + vz * vz;
    Real_v pv      = px * vx + py * vy + pz * vz;
    Real_v distR   = ellipsoid.fQ1 * rr - ellipsoid.fQ2;
    Bool_v outside = vecCore::math::Max(distR, distZ) > kHalfTolerance;

    distance = Real_v(0.);
    vecCore__MaskedAssignFunc(distance, outside, Real_v(-1.));

    // Find intersection with Z planes
    Real_v tzmax = kMaximum;
    vecCore__MaskedAssignFunc(tzmax, vz != Real_v(0.), (vecCore::math::CopySign(Real_v(dzcut), vz) - pzcut) / vz);

    // Find intersection with sphere
    Real_v B     = pv / vv;
    Real_v C     = (rr - ellipsoid.fR * ellipsoid.fR) / vv;
    Real_v D     = B * B - C;
    Real_v sqrtD = vecCore::math::Sqrt(vecCore::math::Abs(D));
    Real_v trmax = -B + sqrtD;

    // Check if no intersection
    Real_v EPS  = Real_v(2.) * rr * vv * kEpsilon;
    Bool_v done = outside || (D <= EPS);

    // Set distance
    vecCore__MaskedAssignFunc(distance, !done, vecCore::math::Min(tzmax, trmax));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &ellipsoid,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the Ellipsoid surface */
    Real_v x = point.x() * ellipsoid.fSx;
    Real_v y = point.y() * ellipsoid.fSy;
    Real_v z = point.z() * ellipsoid.fSz;
    Real_v r = vecCore::math::Sqrt(x * x + y * y + z * z);
    // Set safety to zero if point is on surface
    Real_v safeZ = vecCore::math::Abs(z - ellipsoid.fScZMidCut) - ellipsoid.fScZDimCut;
    Real_v safeR = r - ellipsoid.fR;
    safety       = vecCore::math::Max(safeZ, safeR);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
    // Adjust safety using bounding box
    Real_v distZ  = vecCore::math::Max(point.z() - ellipsoid.fZTopCut, ellipsoid.fZBottomCut - point.z());
    Real_v distXY = vecCore::math::Max(vecCore::math::Abs(point.x()) - ellipsoid.fXmax,
                                       vecCore::math::Abs(point.y()) - ellipsoid.fYmax);
    vecCore__MaskedAssignFunc(safety, safety > Real_v(0.), vecCore::math::Max(safety, distZ, distXY));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &ellipsoid,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the Ellipsoid surface */
    Real_v x = point.x() * ellipsoid.fSx;
    Real_v y = point.y() * ellipsoid.fSy;
    Real_v z = point.z() * ellipsoid.fSz;
    // Set safety to zero if point is on surface
    Real_v safeR = ellipsoid.fR - vecCore::math::Sqrt(x * x + y * y + z * z);
    Real_v safeZ = ellipsoid.fScZDimCut - vecCore::math::Abs(z - ellipsoid.fScZMidCut);
    safety       = vecCore::math::Min(safeZ, safeR);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
    // Adjust safety in z direction
    Real_v distZ = vecCore::math::Min(ellipsoid.fZTopCut - point.z(), point.z() - ellipsoid.fZBottomCut);
    vecCore__MaskedAssignFunc(safety, safety > Real_v(0.), vecCore::math::Min(safeR, distZ));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &ellipsoid, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    //   In case if the point is further than kHalfTolerance from the surface, set valid=false
    //   Must return a valid vector (even if the point is not on the surface)
    //
    //   On an edge provide an average normal of the corresponding base and lateral surface
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v px   = point.x();
    Real_v py   = point.y();
    Real_v pz   = point.z();
    Real_v A    = ellipsoid.fDx;
    Real_v B    = ellipsoid.fDy;
    Real_v C    = ellipsoid.fDz;
    Real_v x    = px * ellipsoid.fSx;
    Real_v y    = py * ellipsoid.fSy;
    Real_v z    = pz * ellipsoid.fSz;
    Real_v mag2 = x * x + y * y + z * z;

    // Check lateral surface
    Real_v distR = ellipsoid.fQ1 * mag2 - ellipsoid.fQ2;
    vecCore__MaskedAssignFunc(normal, vecCore::math::Abs(distR) <= kHalfTolerance,
                              Vector3D<Real_v>(px / (A * A), py / (B * B), pz / (C * C)).Unit());

    // Check z cuts
    Real_v distZ = vecCore::math::Abs(z - ellipsoid.fScZMidCut) - ellipsoid.fScZDimCut;
    vecCore__MaskedAssignFunc(normal[2], vecCore::math::Abs(distZ) <= kHalfTolerance,
                              normal[2] + vecCore::math::Sign(z - ellipsoid.fScZMidCut));

    // Average normal, if required
    vecCore__MaskedAssignFunc(normal, normal.Mag2() > 1., normal.Unit());
    vecCore::Mask_v<Real_v> done = normal.Mag2() > Real_v(0.);
    if (vecCore::MaskFull(done)) return normal;

    // Point is not on the surface - normally, this should never be
    // Return normal to the nearest surface
    vecCore__MaskedAssignFunc(valid, !done, false);
    vecCore__MaskedAssignFunc(normal[2], !done, vecCore::math::Sign(z - ellipsoid.fScZMidCut));
    vecCore__MaskedAssignFunc(distR, !done, vecCore::math::Sqrt(mag2) - ellipsoid.fR);
    vecCore__MaskedAssignFunc(normal, !done && distR > distZ && mag2 > Real_v(0.),
                              Vector3D<Real_v>(px / (A * A), py / (B * B), pz / (C * C)).Unit());
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPSOIDIMPLEMENTATION_H_
