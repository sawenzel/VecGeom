// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file implements the algorithms for EllipticalCone
/// @file volumes/kernel/EllipticalConeImplementation.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/EllipticalConeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>
#include <iomanip>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipticalConeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipticalConeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipticalCone;
template <typename T>
struct EllipticalConeStruct;
class UnplacedEllipticalCone;

struct EllipticalConeImplementation {

  using PlacedShape_t    = PlacedEllipticalCone;
  using UnplacedStruct_t = EllipticalConeStruct<Precision>;
  using UnplacedVolume_t = UnplacedEllipticalCone;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &ellipticalcone,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(ellipticalcone, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &ellipticalcone,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(ellipticalcone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, Bool_v &completelyinside,
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
    Real_v px     = point.x() * ellipticalcone.invDx;
    Real_v py     = point.y() * ellipticalcone.invDy;
    Real_v pz     = point.z();
    Real_v hp     = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v ds     = (hp - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    Real_v dz     = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    Real_v safety = vecCore::math::Max(ds, dz);

    completelyoutside = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &ellipticalcone,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from outside point to the EllipticalCone surface */
    using Bool_v       = vecCore::Mask_v<Real_v>;
    Real_v kTwoEpsilon = 2. * kEpsilon;
    distance           = Real_v(kInfLength);
    Real_v offset(0.);
    Vector3D<Real_v> p(point);

    // Move point closer, if required
    Real_v Rfar2(1024. * ellipticalcone.fRsph * ellipticalcone.fRsph); // 1024 = 32 * 32
    vecCore__MaskedAssignFunc(offset, ((p.Mag2() > Rfar2) && (direction.Dot(p) < Real_v(0.))),
                              p.Mag() - Real_v(2.) * ellipticalcone.fRsph);
    p += offset * direction;

    // Special cases to keep in mind:
    //   0) Point is on the surface and leaving the solid
    //   1) Trajectory is parallel to the surface (A = 0, single root at t = -C/2B)
    //   2) No intersection (D < 0) or touch (D < eps) with lateral surface
    //   3) Exception: when the trajectory traverses the apex (D < eps) and A < 0
    //      then always there is an intersection with the solid

    // Set working variables, transform elliptican cone to cone
    Real_v px  = p.x() * ellipticalcone.invDx;
    Real_v py  = p.y() * ellipticalcone.invDy;
    Real_v pz  = p.z();
    Real_v pz0 = p.z() - ellipticalcone.fDz; // pz if apex would be in origin
    Real_v vx  = direction.x() * ellipticalcone.invDx;
    Real_v vy  = direction.y() * ellipticalcone.invDy;
    Real_v vz  = direction.z();

    // Compute coefficients of the quadratic equation: A t^2 + 2B t + C = 0
    Real_v Ar = vx * vx + vy * vy;
    Real_v Br = px * vx + py * vy;
    Real_v Cr = px * px + py * py;
    // 1) Check if A = 0
    // If so, slightly modify vz to avoid degeneration of the quadratic equation
    // The magnitude of vz will be modified in a way that preserves correct behavior when 0) point is leaving the solid
    Real_v vzvz  = vz * vz;
    Bool_v tinyA = vecCore::math::Abs(Ar - vzvz) < kTwoEpsilon * vzvz;
    vecCore__MaskedAssignFunc(vz, tinyA, vz + vecCore::math::Abs(vz) * kTwoEpsilon);

    Real_v Az = vz * vz;
    Real_v Bz = pz0 * vz;
    Real_v Cz = pz0 * pz0;
    Real_v A  = Ar - Az;
    Real_v B  = Br - Bz;
    Real_v B0 = Br - pz0 * direction.z(); // B calculated with original v.z()
    Real_v C  = Cr - Cz;
    Real_v D  = B * B - A * C;

    // 0) Check if point is leaving the solid
    Real_v sfz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    Real_v nz  = vecCore::math::Sqrt(Cr);
    Real_v sfr = (nz + pz0) * ellipticalcone.cosAxisMin;
    vecCore::MaskedAssign(nz, (vecCore::math::Abs(p.x()) + vecCore::math::Abs(p.y()) < Real_v(0.1) * kHalfTolerance),
                          Real_v(1.));       // point is on z-axis
    Real_v pzA = pz0 + ellipticalcone.dApex; // slightly shifted apex position for "flying away" check
    Bool_v done =
        (sfz >= -kHalfTolerance && pz * vz >= Real_v(0.)) || (sfr >= -kHalfTolerance && Br + nz * vz >= Real_v(0.)) ||
        (pz0 * ellipticalcone.cosAxisMin > -kHalfTolerance && (Cr - pzA * pzA) <= Real_v(0.) && A >= Real_v(0.));

    // 2) Check if scratching (D < eps & A > 0) or no intersection (D < 0)
    // 3) if (D < eps & A < 0) then trajectory traverses the apex area - continue calculation
    vecCore__MaskedAssignFunc(D, (sfr <= Real_v(0.) && D < Real_v(0.)), Real_v(0.));
    done |= (D < Real_v(0.)) || ((D < kTwoEpsilon * B * B) && (A >= Real_v(0.)));

    // Find intersection with Z planes
    Real_v invz  = Real_v(-1.) / NonZero(vz);
    Real_v dz    = vecCore::math::CopySign(Real_v(ellipticalcone.fZCut), invz);
    Real_v tzin  = (pz - dz) * invz;
    Real_v tzout = (pz + dz) * invz;

    // Find roots of the quadratic equation
    Real_v tmp(0.), t1(0.), t2(0.);
    vecCore__MaskedAssignFunc(tmp, !done, -B - vecCore::math::CopySign(vecCore::math::Sqrt(D), B));
    vecCore__MaskedAssignFunc(t1, !done, tmp / A);
    vecCore__MaskedAssignFunc(t2, !done && tmp != 0, C / tmp);
    vecCore__MaskedAssignFunc(t2, !done && tinyA && B != Real_v(0.), -C / (Real_v(2.) * B0)); // A ~ 0, t = -C / 2B
    Real_v tmin = vecCore::math::Min(t1, t2);
    Real_v tmax = vecCore::math::Max(t1, t2);

    // Set default - intersection with lower nappe (A > 0)
    Real_v trin  = tmin;
    Real_v trout = tmax;
    // Check if intersection with upper nappe only, return infinity
    done |= (A >= Real_v(0.) && pz0 + vz * tmin >= Real_v(0.));

    // Check if intersection with both nappes (A < 0)
    vecCore__MaskedAssignFunc(trin, (!done && A < Real_v(0.)), Real_v(-kInfLength));
    vecCore__MaskedAssignFunc(trout, (!done && A < Real_v(0.)), Real_v(kInfLength));
    vecCore__MaskedAssignFunc(trin, (!done && A < Real_v(0.) && vz < Real_v(0.)), tmax);
    vecCore__MaskedAssignFunc(trout, (!done && A < Real_v(0.) && vz > Real_v(0.)), tmin);

    // Set distance
    // No special check for inside points, distance for inside points will be negative
    Real_v tin  = vecCore::math::Max(tzin, trin);
    Real_v tout = vecCore::math::Min(tzout, trout);
    vecCore__MaskedAssignFunc(distance, !done && (tout - tin) >= kHalfTolerance, tin + offset);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &ellipticalcone,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from inside point to the EllipticalCone surface */
    using Bool_v       = vecCore::Mask_v<Real_v>;
    Real_v kTwoEpsilon = 2. * kEpsilon;
    distance           = Real_v(0.);

    // Special cases to keep in mind:
    //   0) Point is on the surface and leaving the solid
    //   1) Trajectory is parallel to the surface (A = 0, single root at t = -C/2B)
    //   2) No intersection (D < 0) or touch (D < eps) with lateral surface
    //   3) Exception: when the trajectory traverses the apex (D < eps) and A < 0
    //      then always there is an intersection with the solid

    // Set working variables, transform elliptican cone to cone
    Real_v px  = point.x() * ellipticalcone.invDx;
    Real_v py  = point.y() * ellipticalcone.invDy;
    Real_v pz  = point.z();
    Real_v pz0 = pz - ellipticalcone.fDz; // pz if apex would be in origin
    Real_v hp  = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v sfr = (hp - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    Real_v sfz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;

    // Check if point is outside
    Bool_v outside = vecCore::math::Max(sfr, sfz) > kHalfTolerance;
    vecCore__MaskedAssignFunc(distance, outside, Real_v(-1.));
    Bool_v done = outside;

    // Compute coefficients of the quadratic equation: A t^2 + 2B t + C = 0
    Real_v vx = direction.x() * ellipticalcone.invDx;
    Real_v vy = direction.y() * ellipticalcone.invDy;
    Real_v vz = direction.z();
    Real_v Ar = vx * vx + vy * vy;
    Real_v Br = px * vx + py * vy;
    Real_v Cr = px * px + py * py;
    // 1) Check if A = 0
    // If so, slightly modify vz to avoid degeneration of the quadratic equation
    // The magnitude of vz will be modified in a way that point is leaving the solid
    Bool_v tinyA = vecCore::math::Abs(Ar - vz * vz) < kTwoEpsilon * vz * vz;
    vecCore__MaskedAssignFunc(vz, tinyA, vz + vecCore::math::Abs(vz) * kTwoEpsilon);

    Real_v Az = vz * vz;
    Real_v Bz = pz0 * vz;
    Real_v Cz = pz0 * pz0;
    Real_v A  = Ar - Az;
    Real_v B  = Br - Bz;
    Real_v B0 = Br - pz0 * direction.z(); // B calculated with original v.z()
    Real_v C  = Cr - Cz;
    Real_v D  = B * B - A * C;
    vecCore__MaskedAssignFunc(D, (sfr <= Real_v(0.) && D < Real_v(0.)), Real_v(0.));

    // 2) Check if scratching (D < eps & A > 0) or no intersection (D < 0)
    // 3) if (D < eps & A < 0) then trajectory traverses the apex area - continue calculation
    done |= (D < Real_v(0.)) || (D < kTwoEpsilon * B * B && A >= Real_v(0.));

    // Find intersection with Z planes
    Real_v tzout = kMaximum;
    vecCore__MaskedAssignFunc(tzout, vz != Real_v(0.),
                              (vecCore::math::CopySign(Real_v(ellipticalcone.fZCut), vz) - pz) / direction.z());

    // Find roots of the quadratic equation
    Real_v tmp(0.), t1(0.), t2(0.);
    vecCore__MaskedAssignFunc(tmp, !done, -B - vecCore::math::CopySign(vecCore::math::Sqrt(D), B));
    vecCore__MaskedAssignFunc(t1, !done, tmp / A);
    vecCore__MaskedAssignFunc(t2, !done && tmp != Real_v(0.), C / tmp);
    vecCore__MaskedAssignFunc(t2, !done && tinyA && B0 != Real_v(0.), -C / (Real_v(2.) * B0)); // A ~ 0, t = -C / 2B
    Real_v tmin = vecCore::math::Min(t1, t2);
    Real_v tmax = vecCore::math::Max(t1, t2);

    // Set default - intersection with lower nappe (A > 0)
    Real_v trout = tmax;
    // Check if intersection with upper nappe only or flying away, return 0
    done |= ((A >= Real_v(0.) && pz0 + vz * tmax >= Real_v(0.)) || (pz0 >= Real_v(0.) && vz >= Real_v(0.)));

    // Check if intersection with both nappes (A < 0)
    vecCore__MaskedAssignFunc(trout, (!done && A < Real_v(0.)), Real_v(kInfLength));
    vecCore__MaskedAssignFunc(trout, (!done && A < Real_v(0.) && vz > Real_v(0.)), tmin);

    // Set distance
    // No special check for inside points, distance for inside points will be negative
    vecCore__MaskedAssignFunc(distance, !done, vecCore::math::Min(tzout, trout));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &ellipticalcone,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the EllipticalCone surface */
    Real_v px = point.x() * ellipticalcone.invDx;
    Real_v py = point.y() * ellipticalcone.invDy;
    Real_v pz = point.z();
    Real_v hp = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v ds = (hp - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    Real_v dz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    safety    = vecCore::math::Max(ds, dz);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &ellipticalcone,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the EllipticalCone surface */
    Real_v px = point.x() * ellipticalcone.invDx;
    Real_v py = point.y() * ellipticalcone.invDy;
    Real_v pz = point.z();
    Real_v hp = vecCore::math::Sqrt(px * px + py * py) + pz;
    Real_v ds = (ellipticalcone.fDz - hp) * ellipticalcone.cosAxisMin;
    Real_v dz = ellipticalcone.fZCut - vecCore::math::Abs(pz);
    safety    = vecCore::math::Min(ds, dz);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    //   In case if the point is further than kHalfTolerance from the surface, set valid=false
    //   Must return a valid vector (even if the point is not on the surface)
    //
    //   On an edge provide an average normal of the corresponding base and lateral surface
    Vector3D<Real_v> normal(0., 0., 0.);
    valid = true;

    // Check z planes
    Real_v px = point.x();
    Real_v py = point.y();
    Real_v pz = point.z();
    Real_v dz = vecCore::math::Abs(pz) - ellipticalcone.fZCut;
    vecCore__MaskedAssignFunc(normal[2], vecCore::math::Abs(dz) <= kHalfTolerance, vecCore::math::Sign(pz));

    // Check lateral surface
    Real_v nx = px * ellipticalcone.invDx * ellipticalcone.invDx;
    Real_v ny = py * ellipticalcone.invDy * ellipticalcone.invDy;
    Real_v nz = vecCore::math::Sqrt(px * nx + py * ny);
    vecCore__MaskedAssignFunc(nz, (nx * nx + ny * ny) == Real_v(0.), Real_v(1.)); // z-axis
    Vector3D<Real_v> nside(nx, ny, nz);
    Real_v ds = (nz + pz - ellipticalcone.fDz) * ellipticalcone.cosAxisMin;
    vecCore__MaskedAssignFunc(normal, vecCore::math::Abs(ds) <= kHalfTolerance, (normal + nside.Unit()).Unit());

    // Check if done
    vecCore::Mask_v<Real_v> done = normal.Mag2() > Real_v(0.);
    if (vecCore::MaskFull(done)) return normal;

    // Point is not on the surface - normally, this should never be
    // Return normal to the nearest surface
    vecCore__MaskedAssignFunc(valid, !done, false);
    vecCore__MaskedAssignFunc(normal[2], !done, vecCore::math::Sign(pz));
    vecCore__MaskedAssignFunc(normal, !done && ds > dz, nside.Unit());
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_
