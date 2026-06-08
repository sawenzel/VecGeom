// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file volumes/kernel/ParaboloidImplementation.h
/// @brief Navigation kernels for the paraboloid solid.
/// @author Marilena Bandieramonte

#ifndef VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/ParaboloidStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ParaboloidImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ParaboloidImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedParaboloid;
template <typename T>
struct ParaboloidStruct;
class UnplacedParaboloid;

/// @brief Implements Paraboloid classification, distance, safety, and normal kernels.
///
/// @details The solid is bounded by the planes `z = -dz` and `z = +dz` and by
/// the surface of revolution `z = a * (x^2 + y^2) + b`. The cached coefficients
/// in `ParaboloidStruct` allow hot predicates to compare `rho^2` with
/// `fK1 * z + fK2`, avoiding square roots for classification and most boundary
/// checks. Distance kernels solve the line/parabola quadratic for side-surface
/// intersections and filter candidate roots by the z extent.
struct ParaboloidImplementation {

  using PlacedShape_t    = PlacedParaboloid;
  using UnplacedStruct_t = ParaboloidStruct<Precision>;
  using UnplacedVolume_t = UnplacedParaboloid;

  /// @brief Tests whether a point is inside or on the paraboloid.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param inside Output flag set to true unless the point is outside.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &paraboloid,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    bool unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, bool, false>(paraboloid, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classifies a point with the paraboloid tolerance band.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param inside Output classification as `EInside`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &paraboloid,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    bool completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, bool, true>(paraboloid, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = Inside_t(EInside::kOutside);
    if (completelyinside) inside = Inside_t(EInside::kInside);
  }

  /// @brief Computes common squared-radius predicates for `Contains` and `Inside`.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param completelyinside Output true for points strictly inside all bounds.
  /// @param completelyoutside Output true for points outside at least one bound.
  /// @tparam ForInside Whether the strict-inside predicate is required.
  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point, Bool_v &completelyinside,
      Bool_v &completelyoutside)
  {
    completelyinside  = Bool_v(false);
    completelyoutside = Bool_v(false);

    Real_v rho2       = point.Perp2();
    Real_v paraRho2   = paraboloid.fK1 * point.z() + paraboloid.fK2;
    Real_v diff       = rho2 - paraRho2;
    Real_v absZ       = Abs(point.z());
    completelyoutside = (absZ > Real_v(paraboloid.fDz + kTolerance)) || (diff > kTolerance);
    if (completelyoutside) return;
    if (ForInside) completelyinside = (absZ < Real_v(paraboloid.fDz - kTolerance)) && (diff < -kTolerance);
  }

  /// @brief Tests whether a point is on one of the z-plane caps.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @return True when the point is within z tolerance and inside the selected cap radius.
  /// @tparam ForTopZPlane Selects the `+dz` cap when true, otherwise the `-dz` cap.
  template <typename Real_v, bool ForTopZPlane>
  VECCORE_ATT_HOST_DEVICE static bool IsOnZPlane(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point)
  {
    Real_v rho2 = point.Perp2();
    if (ForTopZPlane) {
      return Abs(point.z() - paraboloid.fDz) < kTolerance && rho2 < (paraboloid.fRhi2 + kHalfTolerance);
    } else {
      return Abs(point.z() + paraboloid.fDz) < kTolerance && rho2 < (paraboloid.fRlo2 + kHalfTolerance);
    }
  }

  /// @brief Tests whether a point is on the parabolic side surface.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @return True when the side-surface implicit equation is within tolerance.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE static bool IsOnParabolicSurface(UnplacedStruct_t const &paraboloid,
                                                           Vector3D<Real_v> const &point)
  {
    Real_v value = paraboloid.fA * point.Perp2() + paraboloid.fB - point.z();
    return value > -kTolerance && value < kTolerance;
  }

  /// @brief Computes the first distance from an outside point into the paraboloid.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param direction Unit direction in the local solid frame.
  /// @param stepMax Unused interface parameter.
  /// @param distance Output distance, `kInfLength` for a miss, or `-1` for an inside start.
  ///
  /// @details The kernel handles boundary starts before solving intersections,
  /// then tests cap crossings and the quadratic side-surface intersection. Very
  /// far starting points may be translated closer along the ray before solving,
  /// with the translation added back to accepted distances.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &paraboloid,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /* stepMax */, Real_v &distance)
  {

    bool done(false);
    distance = InfinityLength<Real_v>();
    Real_v offset(0.);
    Vector3D<Real_v> p(point);

    // Move point closer, if required
    Precision Rsph = 1.5 * vecCore::math::Max(paraboloid.fDx, paraboloid.fDz);
    Real_v Rfar2(1024. * Rsph * Rsph); // 1024 = 32 * 32
    if ((p.Mag2() > Rfar2) && (direction.Dot(p) < Real_v(0.))) offset = p.Mag() - Real_v(2.) * Rsph;
    p += offset * direction;

    Real_v absZ = Abs(p.z());
    Real_v rho2 = p.Perp2(); // p.x()*p.x()+p.y()*p.y();
    bool checkZ = p.z() * direction.z() >= Real_v(0.);

    // check if the point is distancing in Z
    bool isDistancingInZ = (absZ > paraboloid.fDz && checkZ);
    done |= isDistancingInZ;
    if (done) return;

    Real_v paraRho2 = paraboloid.fK1 * p.z() + paraboloid.fK2;
    Real_v diff     = rho2 - paraRho2;

    distance                                  = Real_v(-1.);
    bool insideZ                              = absZ < Real_v(paraboloid.fDz - kTolerance);
    bool insideParabolicSurfaceOuterTolerance = (diff < -kTolerance);
    done |= !done && (insideZ && insideParabolicSurfaceOuterTolerance);
    if (done) return;

    bool isOnZPlaneAndMovingInside = (IsOnZPlane<Real_v, true>(paraboloid, point) && direction.z() < Real_v(0.)) ||
                                     (IsOnZPlane<Real_v, false>(paraboloid, point) && direction.z() > Real_v(0.));
    if (!done && isOnZPlaneAndMovingInside) distance = Real_v(0.);
    done |= isOnZPlaneAndMovingInside;
    if (done) return;

    Vector3D<Real_v> normal(p.x(), p.y(), Real_v(-paraboloid.fK1 * Real_v(0.5)));
    Real_v normalDotDirection = direction.Dot(normal);
    bool isOnParabolicSurfaceAndMovingInside =
        diff > -kTolerance && diff < kTolerance && normalDotDirection < -Real_v(0.5 * kTolerance) * direction.Perp2();
    if (!done && isOnParabolicSurfaceAndMovingInside) distance = Real_v(0.);
    done |= isOnParabolicSurfaceAndMovingInside;
    if (done) return;

    distance = InfinityLength<Real_v>();

    /* Intersection tests with Z planes are not required if the point is within Z range.
     * In this case it will either intersect with the parabolic surface or not intersect at all.
     */
    if (!(absZ < paraboloid.fDz)) {
      Real_v distZ(InfinityLength<Real_v>());                          // = (absZ - paraboloid.fDz) / absDirZ;
      bool bottomPlane = p.z() < -paraboloid.fDz && direction.z() > 0; //(true);
      bool topPlane    = p.z() > paraboloid.fDz && direction.z() < 0;
      if (topPlane) distZ = (paraboloid.fDz - p.z()) / NonZero(direction.z());
      if (bottomPlane) distZ = (-paraboloid.fDz - p.z()) / NonZero(direction.z());
      Real_v xHit    = p.x() + distZ * direction.x();
      Real_v yHit    = p.y() + distZ * direction.y();
      Real_v rhoHit2 = xHit * xHit + yHit * yHit;

      if (!done && topPlane && rhoHit2 <= paraboloid.fRhi2) distance = distZ + offset;
      done |= topPlane && rhoHit2 < paraboloid.fRhi2;
      if (done) return;

      if (!done && bottomPlane && rhoHit2 <= paraboloid.fRlo2) distance = distZ + offset;
      done |= (bottomPlane && rhoHit2 <= paraboloid.fRlo2); // || (topPlane && rhoHit2 <= paraboloid.fRhi2);
      if (done) return;
    }

    /* Intersection tests with Parabolic surface are not required if the point is above
     * top Z plane Radius of point is less the Rhi. In this case depending upon the
     * direction it will either intersect with top Z plane or not intersect at all
     */
    if (!(p.z() > paraboloid.fDz && rho2 < paraboloid.fRhi2)) {
      // Quadratic Solver for Parabolic surface
      Real_v dirRho2 = direction.Perp2();
      Real_v pDotV2D = p.x() * direction.x() + p.y() * direction.y();
      Real_v a       = paraboloid.fA * dirRho2;
      Real_v b       = Real_v(0.5) * direction.z() - paraboloid.fA * pDotV2D;
      Real_v c       = (paraboloid.fB + paraboloid.fA * rho2 - p.z());
      Real_v d2      = b * b - a * c;
      done |= d2 < Real_v(0.);
      if (done) return;

      Real_v distParab = InfinityLength<Real_v>();
      Real_v sqrtD     = Sqrt(d2);
      if (b <= Real_v(0.)) {
        distParab = (b - sqrtD) / NonZero(a);
      } else {
        distParab = (c / NonZero(b + sqrtD));
      }
      Real_v zHit = p.z() + distParab * direction.z();
      if (Abs(zHit) <= paraboloid.fDz && distParab > Real_v(0.)) distance = distParab + offset;
    }
  }

  /// @brief Computes the first distance from an inside point out of the paraboloid.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param direction Unit direction in the local solid frame.
  /// @param stepMax Unused interface parameter.
  /// @param distance Output distance, or `-1` for an outside start.
  ///
  /// @details Candidate exits are the nearest z-plane cap crossing and the
  /// positive side-surface quadratic root. Boundary starts moving through the
  /// selected surface return zero before the general root solve.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &paraboloid,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {

    // setting distance to -1. for wrong side points
    distance = -1.;
    bool done(false);

    // Outside Z range
    bool outsideZ = Abs(point.z()) > paraboloid.fDz + kTolerance;
    done |= outsideZ;
    if (done) return;

    // Outside Parabolic surface
    Real_v rho2                                = point.Perp2();
    Real_v paraRho2                            = paraboloid.fK1 * point.z() + paraboloid.fK2;
    Real_v value                               = rho2 - paraRho2;
    bool outsideParabolicSurfaceOuterTolerance = (value > kHalfTolerance);
    done |= outsideParabolicSurfaceOuterTolerance;
    if (done) return;

    // On Z Plane and moving outside;
    bool isOnZPlaneAndMovingOutside = (IsOnZPlane<Real_v, true>(paraboloid, point) && direction.z() > Real_v(0.)) ||
                                      (IsOnZPlane<Real_v, false>(paraboloid, point) && direction.z() < Real_v(0.));
    if (!done && isOnZPlaneAndMovingOutside) distance = Real_v(0.);
    done |= isOnZPlaneAndMovingOutside;
    if (done) return;

    // On Parabolic Surface and moving outside
    Vector3D<Real_v> normal(point.x(), point.y(), Real_v(-paraboloid.fK1 * Real_v(0.5)));
    bool isOnParabolicSurfaceAndMovingInside =
        value > -kTolerance && value < kTolerance && direction.Dot(normal) > Real_v(0.);
    if (!done && isOnParabolicSurfaceAndMovingInside) distance = Real_v(0.);
    done |= isOnParabolicSurfaceAndMovingInside;
    if (done) return;

    distance = InfinityLength<Real_v>();

    Real_v distZ   = InfinityLength<Real_v>();
    Real_v dirZinv = Real_v(1.) / NonZero(direction.z());

    bool dir_mask = direction.z() < 0;
    if (dir_mask) {
      distZ = -(paraboloid.fDz + point.z()) * dirZinv;
    } else {
      distZ = (paraboloid.fDz - point.z()) * dirZinv;
    }

    Real_v dirRho2 = direction.Perp2();
    Real_v pDotV2D = point.x() * direction.x() + point.y() * direction.y();
    Real_v a       = Real_v(paraboloid.fA * dirRho2);
    Real_v b       = Real_v(0.5) * direction.z() - Real_v(paraboloid.fA) * pDotV2D;
    Real_v c       = paraboloid.fB + paraboloid.fA * rho2 - point.z();
    Real_v d2      = b * b - a * c;

    Real_v distParab = InfinityLength<Real_v>();
    if (d2 >= Real_v(0.) && (b > Real_v(0.))) {
      distParab = (b + Sqrt(d2)) * (Real_v(1.) / NonZero(a));
    }
    if (d2 >= Real_v(0.) && (b <= Real_v(0.))) distParab = (c / NonZero(b - Sqrt(d2)));
    distance = Min(distParab, distZ);
  }

  /// @brief Estimates safety from an outside point to the paraboloid.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param safety Output safety, `-1` for an inside start, or zero on a boundary.
  ///
  /// @details The estimate combines the z-plane separation with a tangent-based
  /// side-surface estimate derived from the local parabola slope.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &paraboloid,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {

    Real_v absZ  = Abs(point.z());
    Real_v safeZ = absZ - paraboloid.fDz;

    safety = -1.;
    bool done(false);
    bool insideZ = absZ < paraboloid.fDz - kTolerance;

    Real_v rho2                               = point.Perp2();
    Real_v value                              = paraboloid.fA * rho2 + paraboloid.fB - point.z();
    bool insideParabolicSurfaceOuterTolerance = (value < -kHalfTolerance);
    done |= (insideZ && insideParabolicSurfaceOuterTolerance);
    if (done) return;

    bool onZPlane =
        Abs(Abs(point.z()) - paraboloid.fDz) < kTolerance &&
        (rho2 < Real_v(paraboloid.fRhi2 + kHalfTolerance) || rho2 < Real_v(paraboloid.fRlo2 + kHalfTolerance));
    if (onZPlane) safety = Real_v(0.);
    done |= onZPlane;
    if (done) return;

    bool onParabolicSurface = value > -kTolerance && value < kTolerance;
    if (!done && onParabolicSurface) safety = Real_v(0.);
    done |= onParabolicSurface;
    if (done) return;

    safety = InfinityLength<Real_v>();

    Real_v r0sq = (point.z() - paraboloid.fB) * paraboloid.fInvA;

    safety = safeZ;

    bool underParaboloid = (r0sq < 0);
    done |= underParaboloid;
    if (done) return;

    Real_v safeR = InfinityLength<Real_v>();
    Real_v ro2   = point.x() * point.x() + point.y() * point.y();
    Real_v r0    = Sqrt(r0sq);
    Real_v dr    = Sqrt(ro2) - r0;

    bool drCloseToZero = (dr < Real_v(1.E-8));
    done |= drCloseToZero;
    if (done) return;

    // Use the local tangent to estimate the side-surface safety.
    Real_v talf = Real_v(-2.) * paraboloid.fA * r0;
    Real_v salf = talf / Sqrt(Real_v(1.) + talf * talf);
    safeR       = Abs(dr * salf);

    Real_v max_safety = Max(safeR, safeZ);
    if (!done) safety = max_safety;
  }

  /// @brief Estimates safety from an inside point to the paraboloid boundary.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param safety Output safety, `-1` for an outside start, or zero on a boundary.
  ///
  /// @details The side-surface contribution is computed from the radial
  /// separation to the parabola at the point's z coordinate and is combined with
  /// the z-plane safety.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &paraboloid,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {

    Real_v absZ = Abs(point.z());
    Real_v safZ = (paraboloid.fDz - absZ);

    safety = -1.;
    bool done(false);
    bool outsideZ = absZ > Real_v(paraboloid.fDz + kTolerance);
    done |= outsideZ;
    if (done) return;

    Real_v rho2                                = point.Perp2();
    Real_v value                               = paraboloid.fA * rho2 + paraboloid.fB - point.z();
    bool outsideParabolicSurfaceOuterTolerance = (value > kHalfTolerance);
    done |= outsideParabolicSurfaceOuterTolerance;
    if (done) return;

    bool onZPlane =
        Abs(Abs(point.z()) - paraboloid.fDz) < kTolerance &&
        (rho2 < Real_v(paraboloid.fRhi2 + kHalfTolerance) || rho2 < Real_v(paraboloid.fRlo2 + kHalfTolerance));
    if (onZPlane) safety = Real_v(0.);
    done |= onZPlane;
    if (done) return;

    bool onSurface = value > -kTolerance && value < kTolerance;
    if (!done && onSurface) safety = Real_v(0.);
    done |= onSurface;
    if (done) return;
    Real_v r0sq = (point.z() - paraboloid.fB) * paraboloid.fInvA;

    safety = 0.;

    bool closeToParaboloid = (r0sq < 0);
    done |= closeToParaboloid;
    if (done) return;

    Real_v safR = InfinityLength<Real_v>();
    Real_v ro2  = point.x() * point.x() + point.y() * point.y();
    Real_v z0   = paraboloid.fA * ro2 + paraboloid.fB;
    Real_v dr   = Sqrt(ro2) - Sqrt(r0sq); // avoid square root of a negative number

    bool drCloseToZero = (dr > Real_v(-1.E-8));
    done |= drCloseToZero;
    if (done) return;

    Real_v dz = Abs(point.z() - z0);
    safR      = -dr * dz / Sqrt(dr * dr + dz * dz);

    Real_v min_safety = Min(safR, safZ);
    if (!done) safety = min_safety;
  }

  /// @brief Computes the outward normal or an approximate normal.
  /// @param paraboloid Cached unplaced paraboloid data.
  /// @param point Point in the local solid frame.
  /// @param valid Output flag set when the point is on a recognized surface.
  /// @return Unit normal for surface points, or an approximate unit normal otherwise.
  ///
  /// @details Surface normals may combine a z-plane cap normal and the parabolic
  /// side normal at cap/side edges. The parabolic normal is constructed only
  /// when needed; axis points use the side normal limit.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    Vector3D<Real_v> normal(0., 0., 0.);
    Real_v nsurf(0.);

    Real_v rho2           = point.Perp2();
    bool isOnTopZPlane    = Abs(point.z() - paraboloid.fDz) < kTolerance && rho2 < (paraboloid.fRhi2 + kHalfTolerance);
    bool isOnBottomZPlane = Abs(point.z() + paraboloid.fDz) < kTolerance && rho2 < (paraboloid.fRlo2 + kHalfTolerance);
    bool isOnZPlane       = isOnTopZPlane || isOnBottomZPlane;
    Real_v surfaceValue   = paraboloid.fA * rho2 + paraboloid.fB - point.z();
    bool isOnParabolicSurface = surfaceValue > -kTolerance && surfaceValue < kTolerance;

    if (isOnZPlane) nsurf += 1;
    if (isOnTopZPlane) normal[2] = Real_v(1.);
    if (isOnBottomZPlane) normal[2] = Real_v(-1.);

    if (isOnParabolicSurface) {
      // The interface is scalar, so treat points on the axis of symmetry explicitly.
      Vector3D<Real_v> normParabolic(0., 0., vecCore::math::Sign(-paraboloid.fA));
      Real_v r = Sqrt(rho2);
      if (r > kTolerance) {
        Real_v talf   = -2 * paraboloid.fA * r;
        Real_v calf   = 1. / Sqrt(1. + talf * talf);
        Real_v salf   = talf * calf;
        normParabolic = Vector3D<Real_v>((salf * point.x() / NonZero(r)), (salf * point.y() / NonZero(r)), calf);
      }
      nsurf += 1;
      normal[0] -= normParabolic[0];
      normal[1] -= normParabolic[1];
      normal[2] -= normParabolic[2];
    }

    valid = (nsurf > 0);

    if (valid) return normal.Normalized();

    Vector3D<Real_v> norm(0., 0., 0.);
    if (point.z() > Real_v(0.)) norm[2] = Real_v(1.);
    if (point.z() < Real_v(0.)) norm[2] = Real_v(-1.);

    Real_v r    = Sqrt(rho2);
    Real_v safz = paraboloid.fDz - Abs(point.z());
    Real_v safr = Abs(r - Sqrt((point.z() - paraboloid.fB) * paraboloid.fInvA));
    if (safz >= Real_v(0.) && safr < safz) {
      Vector3D<Real_v> normParabolic(0., 0., vecCore::math::Sign(-paraboloid.fA));
      if (r > kTolerance) {
        Real_v talf   = -2 * paraboloid.fA * r;
        Real_v calf   = 1. / Sqrt(1. + talf * talf);
        Real_v salf   = talf * calf;
        normParabolic = Vector3D<Real_v>((salf * point.x() / NonZero(r)), (salf * point.y() / NonZero(r)), calf);
      }
      norm = normParabolic;
    }

    if (!valid) normal = norm;

    return normal.Normalized();
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_
