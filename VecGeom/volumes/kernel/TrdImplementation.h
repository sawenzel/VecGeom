// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file implements the algorithms for Trd
/// @file volumes/kernel/TrdImplementation.h
/// @author Georgios Bitzes

#ifndef VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/TrdStruct.h"
#include "VecGeom/volumes/kernel/shapetypes/TrdTypes.h"
#include <stdlib.h>
#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, TrdImplementation, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace TrdUtilities {

/// @brief Computes the signed orientation of a point relative to a line through the origin.
/// @details This evaluates the 2D cross product `v x p = vx * py - vy * px`
/// for the directed line vector `v = (vx, vy)` and the point vector
/// `p = (px, py)`. With the usual x-right/y-up convention, a positive result
/// means the point is on the CCW/left side of the directed line and a negative
/// result means it is on the CW/right side. The Trd inside checks pass
/// already-shifted point coordinates so this origin-line test applies to each
/// lateral face.
/// @tparam Real_v Scalar or vector floating-point type used for point coordinates.
/// @param px Point x coordinate in the line-local frame.
/// @param py Point y coordinate in the line-local frame.
/// @param vx Line direction x component.
/// @param vy Line direction y component.
/// @param crossProduct Output cross product `vx * py - vy * px`.
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PointLineOrientation(Real_v const &px, Real_v const &py,
                                                                       Precision const &vx, Precision const &vy,
                                                                       Real_v &crossProduct)
{
  crossProduct = vx * py - vy * px;
}

/// @brief Intersects a ray with one bounded lateral Trd face.
/// @details The face is first treated as its infinite supporting plane in the
/// `(V, Z)` section, where `V` is either `X` or `Y`. Points on the lateral edge
/// satisfy `(v1, -dz) + s * (alongV, alongZ)`, while ray points satisfy
/// `(posV, posZ) + t * (dirV, dirZ)`. Solving these two equations for the ray
/// parameter gives
/// `t = (alongZ * (posV - v1) - alongV * (posZ + dz)) /
///      (dirZ * alongV - dirV * alongZ)`.
///
/// The resulting `t` is only a hit on the bounded Trd face if the point also
/// lies within the finite Z range and within the varying half-width in the
/// orthogonal `K` coordinate. Mirrored faces are handled by flipping `V` and
/// `dirV` before applying the same formula.
/// @tparam Real_v Scalar or vector floating-point type.
/// @tparam forY Selects a Y-varying face when true, otherwise an X-varying face.
/// @tparam mirroredPoint Selects the mirrored face of the same axis.
/// @tparam toInside Selects the orientation convention for DistanceToIn versus DistanceToOut.
/// @param trd Trd data structure with cached geometric coefficients.
/// @param pos Ray origin in the Trd local frame.
/// @param dir Unit ray direction in the Trd local frame.
/// @param dist Output distance to the candidate face when the function returns true.
/// @return True if the ray hits the bounded face with the requested orientation.
template <typename Real_v, bool forY, bool mirroredPoint, bool toInside>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool FaceTrajectoryIntersection(TrdStruct<Precision> const &trd,
                                                                             Vector3D<Real_v> const &pos,
                                                                             Vector3D<Real_v> const &dir, Real_v &dist)
{
  Real_v alongV, posV, dirV, posK, dirK, fV, fK, halfKplus, v1;
  if (forY) {
    alongV    = trd.fY2minusY1;
    v1        = trd.fDY1;
    posV      = pos.y();
    posK      = pos.x();
    dirV      = dir.y();
    dirK      = dir.x();
    fK        = trd.fFx;
    fV        = trd.fFy;
    halfKplus = trd.fHalfX1plusX2;
  } else {
    alongV    = trd.fX2minusX1;
    v1        = trd.fDX1;
    posV      = pos.x();
    posK      = pos.y();
    dirV      = dir.x();
    dirK      = dir.y();
    fK        = trd.fFy;
    fV        = trd.fFx;
    halfKplus = trd.fHalfY1plusY2;
  }
  if (mirroredPoint) {
    posV *= Real_v(-1.);
    dirV *= Real_v(-1.);
  }

  Real_v alongZ       = Real_v(2.0) * trd.fDZ;
  Real_v ndotv_alongZ = alongZ * (dirV + fV * dir.z());
  bool ok             = toInside ? ndotv_alongZ < -kTolerance : ndotv_alongZ > kTolerance;
  if (!ok) return false;

  // distance from trajectory to face
  dist = (alongZ * (posV - v1) - alongV * (pos.z() + trd.fDZ)) / (dir.z() * alongV - dirV * alongZ + kTiny);
  if (dist <= Real_v(MakeMinusTolerant<true>(0.))) return false;

  // Validate that the candidate hits the bounded trapezoid face, not only its
  // infinite supporting plane.
  Real_v hitz = pos.z() + dist * dir.z();
  if (vecCore::math::Abs(hitz) >= MakePlusTolerant<true>(trd.fDZ)) return false;

  Real_v hitk = posK + dist * dirK;
  Real_v dK   = halfKplus - fK * hitz; // width of the varying dimension at hitz
  if (vecCore::math::Abs(hitk) >= MakePlusTolerant<true>(dK)) return false;

  if (vecCore::math::Abs(dist) < kHalfTolerance) dist = Real_v(0.0);
  return true;
}

/// @brief Computes signed Trd safety from either inside or outside.
/// @tparam Real_v Scalar or vector floating-point type.
/// @tparam trdTypeT Trd specialization tag controlling whether Y varies with Z.
/// @tparam inside True for SafetyToOut sign convention, false for SafetyToIn.
/// @param trd Trd data structure with cached geometric coefficients.
/// @param pos Query point in the Trd local frame.
/// @param dist Output signed safety according to the selected convention.
template <typename Real_v, typename trdTypeT, bool inside>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Safety(TrdStruct<Precision> const &trd, Vector3D<Real_v> const &pos,
                                                         Real_v &dist)
{
  using namespace TrdTypes;

  Real_v safz = trd.fDZ - vecCore::math::Abs(pos.z());
  dist        = safz;

  Real_v distx = trd.fHalfX1plusX2 - trd.fFx * pos.z();
  Real_v safx  = (distx - vecCore::math::Abs(pos.x())) * trd.fCalfX;
  if (distx >= 0 && safx < dist) dist = safx;

  if (checkVaryingY<trdTypeT>(trd)) {
    Real_v disty = trd.fHalfY1plusY2 - trd.fFy * pos.z();
    Real_v safy  = (disty - vecCore::math::Abs(pos.y())) * trd.fCalfY;
    if (disty >= 0 && safy < dist) dist = safy;
  } else {
    Real_v safy = trd.fDY1 - vecCore::math::Abs(pos.y());
    if (safy < dist) dist = safy;
  }
  if (!inside) dist = -dist;
}

/// @brief Classifies a local point against the unplaced Trd.
/// @tparam Real_v Scalar or vector floating-point type.
/// @tparam trdTypeT Trd specialization tag controlling whether Y varies with Z.
/// @tparam surfaceT Enables tolerance bands and inside/surface distinction when true.
/// @param trd Trd data structure with cached geometric coefficients.
/// @param point Query point in the Trd local frame.
/// @param completelyinside Set true only when the point is strictly inside all relevant planes.
/// @param completelyoutside Set true when the point is outside at least one relevant plane.
template <typename Real_v, typename trdTypeT, bool surfaceT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedInside(TrdStruct<Precision> const &trd,
                                                                        Vector3D<Real_v> const &point,
                                                                        bool &completelyinside, bool &completelyoutside)
{

  using namespace TrdUtilities;
  using namespace TrdTypes;

  Real_v pzPlusDz = point.z() + trd.fDZ;

  // inside Z?
  completelyoutside = vecCore::math::Abs(point.z()) > MakePlusTolerant<surfaceT>(trd.fDZ);
  completelyinside  = surfaceT ? vecCore::math::Abs(point.z()) < MakeMinusTolerant<surfaceT>(trd.fDZ) : false;

  // inside X?
  Real_v cross;
  // Note: we cannot compare directly the cross product with the surface tolerance, but with
  // the tolerance multiplied by the length of the lateral segment connecting dx1 and dx2
  PointLineOrientation<Real_v>(vecCore::math::Abs(point.x()) - trd.fDX1, pzPlusDz, trd.fX2minusX1, 2.0 * trd.fDZ,
                               cross);
  if (surfaceT) {
    completelyoutside |= cross < -trd.fToleranceX;
    completelyinside &= cross > trd.fToleranceX;
  } else {
    completelyoutside |= cross < 0;
  }

  // inside Y?
  if (HasVaryingY<trdTypeT>::value != TrdTypes::kNo) {
    // If Trd type is unknown don't bother with a runtime check, assume the general case
    PointLineOrientation<Real_v>(vecCore::math::Abs(point.y()) - trd.fDY1, pzPlusDz, trd.fY2minusY1, 2.0 * trd.fDZ,
                                 cross);
    if (surfaceT) {
      completelyoutside |= cross < -trd.fToleranceY;
      completelyinside &= cross > trd.fToleranceY;
    } else {
      completelyoutside |= cross < 0;
    }
  } else {
    completelyoutside |= vecCore::math::Abs(point.y()) > MakePlusTolerant<surfaceT>(trd.fDY1);
    if (surfaceT) completelyinside &= vecCore::math::Abs(point.y()) < MakeMinusTolerant<surfaceT>(trd.fDY1);
  }
}

} // namespace TrdUtilities

template <typename T>
class SPlacedTrd;
template <typename T>
class SUnplacedTrd;

template <typename T>
struct TrdStruct;

/// @brief Kernel implementation for Trd point classification, distances, and safeties.
/// @tparam trdTypeT Trd specialization tag controlling whether Y varies with Z.
template <typename trdTypeT>
struct TrdImplementation {

  using UnplacedStruct_t = TrdStruct<Precision>;
  using UnplacedVolume_t = SUnplacedTrd<trdTypeT>;
  using PlacedShape_t    = SPlacedTrd<UnplacedVolume_t>;

  /// @brief Tests whether a local point is inside or on the unplaced Trd.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @tparam Bool_v Boolean result type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Query point in the Trd local frame.
  /// @param inside Output containment flag, true for inside or surface points.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContains(UnplacedStruct_t const &trd,
                                                                            Vector3D<Real_v> const &point,
                                                                            Bool_v &inside)
  {

    bool unused(false);
    bool outside(false);
    TrdUtilities::UnplacedInside<Real_v, trdTypeT, false>(trd, point, unused, outside);
    inside = !outside;
  }

  /// @brief Tests whether a local point is inside or on the placed Trd.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @tparam Bool_v Boolean result type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Query point in the Trd local frame.
  /// @param inside Output containment flag, true for inside or surface points.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &trd,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {

    bool unused(false);
    bool outside(false);
    TrdUtilities::UnplacedInside<Real_v, trdTypeT, false>(trd, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classifies a local point as inside, outside, or surface.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @tparam Inside_v Integral inside-code result type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Query point in the Trd local frame.
  /// @param inside Output classification using `EInside` values.
  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &trd,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    bool inmask(false);
    bool outmask(false);

    TrdUtilities::UnplacedInside<Real_v, trdTypeT, true>(trd, point, inmask, outmask);

    inside = outmask ? EInside::kOutside : (inmask ? EInside::kInside : EInside::kSurface);
  }

  /// @brief Computes distance from an outside point to the Trd boundary along a ray.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Ray origin in the Trd local frame.
  /// @param direction Unit ray direction in the Trd local frame.
  /// @param distance Output distance to enter, `-1` for points already inside, or infinity for no hit.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &trd,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {

    using namespace TrdUtilities;
    using namespace TrdTypes;

    Real_v hitx, hity;
    // Real_v hitz;

    distance = InfinityLength<Real_v>();

    // hit Z faces?
    bool inz     = vecCore::math::Abs(point.z()) < Real_v(MakeMinusTolerant<true>(trd.fDZ));
    Real_v distx = trd.fHalfX1plusX2 - trd.fFx * point.z();
    bool inx     = (distx - vecCore::math::Abs(point.x())) * trd.fCalfX > Real_v(MakePlusTolerant<true>(0.));
    Real_v disty;
    bool iny;
    if (checkVaryingY<trdTypeT>(trd)) {
      disty = trd.fHalfY1plusY2 - trd.fFy * point.z();
      iny   = (disty - vecCore::math::Abs(point.y())) * trd.fCalfY > Real_v(MakePlusTolerant<true>(0.));
    } else {
      disty = vecCore::math::Abs(point.y()) - trd.fDY1;
      iny   = disty < Real_v(MakeMinusTolerant<true>(0.));
    }
    if (inx && iny && inz) {
      distance = Real_v(-1.);
      return;
    }

    bool okz = point.z() * direction.z() < Real_v(0.);
    okz &= !inz;
    if (okz) {
      Real_v distz = (vecCore::math::Abs(point.z()) - trd.fDZ) / vecCore::math::Abs(direction.z());
      // exclude case in which particle is going away
      hitx = vecCore::math::Abs(point.x() + distz * direction.x());
      hity = vecCore::math::Abs(point.y() + distz * direction.y());

      // hitting top face?
      bool okzt = point.z() > (trd.fDZ - kHalfTolerance) && hitx <= trd.fDX2 && hity <= trd.fDY2;
      // hitting bottom face?
      bool okzb = point.z() < (-trd.fDZ + kHalfTolerance) && hitx <= trd.fDX1 && hity <= trd.fDY1;

      okz &= (okzt | okzb);
      if (okz) {
        distance = distz;
        if (vecCore::math::Abs(distance) < kHalfTolerance) distance = Real_v(0.0);
        return;
      }
    }

    // hitting X faces?
    if (!inx) {

      if (FaceTrajectoryIntersection<Real_v, false, false, true>(trd, point, direction, distx)) {
        distance = distx;
        return;
      }

      if (FaceTrajectoryIntersection<Real_v, false, true, true>(trd, point, direction, distx)) {
        distance = distx;
        return;
      }
    }

    // hitting Y faces?
    if (checkVaryingY<trdTypeT>(trd)) {
      if (!iny) {
        if (FaceTrajectoryIntersection<Real_v, true, false, true>(trd, point, direction, disty)) {
          distance = disty;
          return;
        }

        if (FaceTrajectoryIntersection<Real_v, true, true, true>(trd, point, direction, disty)) {
          distance = disty;
          return;
        }
      }
    } else {
      if (!iny) {
        disty /= vecCore::math::Abs(direction.y());
        Real_v zhit = point.z() + disty * direction.z();
        Real_v xhit = point.x() + disty * direction.x();
        Real_v dx   = trd.fHalfX1plusX2 - trd.fFx * zhit;
        bool oky    = point.y() * direction.y() < 0 && disty > -kHalfTolerance && vecCore::math::Abs(xhit) < dx &&
                   vecCore::math::Abs(zhit) < trd.fDZ;
        if (oky) distance = disty;
      }
    }
    if (vecCore::math::Abs(distance) < kHalfTolerance) distance = Real_v(0.0);
  }

  /// @brief Computes distance from an inside point to the Trd boundary along a ray.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Ray origin in the Trd local frame.
  /// @param dir Unit ray direction in the Trd local frame.
  /// @param distance Output distance to exit, or `-1` for points already outside.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &trd,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const & /*stepMax*/, Real_v &distance)
  {

    using namespace TrdUtilities;
    using namespace TrdTypes;

    Real_v hitx, hity;
    // Real_v hitz;
    distance = Real_v(0.0);

    // hit top Z face?
    Real_v invdir = Real_v(1.) / vecCore::math::Abs(dir.z() + kTiny);
    Real_v safz   = trd.fDZ - vecCore::math::Abs(point.z());
    bool out      = safz < Real_v(MakeMinusTolerant<true>(0.));
    Real_v distx  = trd.fHalfX1plusX2 - trd.fFx * point.z();
    out |= (distx - vecCore::math::Abs(point.x())) * trd.fCalfX < Real_v(MakeMinusTolerant<true>(0.));
    Real_v disty;
    if (checkVaryingY<trdTypeT>(trd)) {
      disty = trd.fHalfY1plusY2 - trd.fFy * point.z();
      out |= (disty - vecCore::math::Abs(point.y())) * trd.fCalfY < Real_v(MakeMinusTolerant<true>(0.));
    } else {
      disty = trd.fDY1 - vecCore::math::Abs(point.y());
      out |= disty < Real_v(MakeMinusTolerant<true>(0.));
    }
    if (out) {
      distance = Real_v(-1.);
      return;
    }
    auto maxXY = Max(trd.fHalfX1plusX2, trd.fHalfY1plusY2);
    bool okzt  = dir.z() * maxXY > kTolerance;
    if (okzt) {
      Real_v distz = (trd.fDZ - point.z()) * invdir;
      hitx         = vecCore::math::Abs(point.x() + distz * dir.x());
      hity         = vecCore::math::Abs(point.y() + distz * dir.y());
      okzt &= hitx < MakePlusTolerant<true>(trd.fDX2) && hity < MakePlusTolerant<true>(trd.fDY2);
      if (okzt) {
        distance = distz;
        if (vecCore::math::Abs(distance) < kHalfTolerance) distance = Real_v(0.0);
        return;
      }
    }

    // hit bottom Z face?
    bool okzb = dir.z() * maxXY < -kTolerance;
    if (okzb) {
      Real_v distz = (point.z() + trd.fDZ) * invdir;
      hitx         = vecCore::math::Abs(point.x() + distz * dir.x());
      hity         = vecCore::math::Abs(point.y() + distz * dir.y());
      okzb &= hitx < MakePlusTolerant<true>(trd.fDX1) && hity < MakePlusTolerant<true>(trd.fDY1);
      if (okzb) {
        distance = distz;
        if (vecCore::math::Abs(distance) < kHalfTolerance) distance = Real_v(0.0);
        return;
      }
    }

    // hitting X faces?
    if (FaceTrajectoryIntersection<Real_v, false, false, false>(trd, point, dir, distx)) {
      distance = distx;
      return;
    }

    if (FaceTrajectoryIntersection<Real_v, false, true, false>(trd, point, dir, distx)) {
      distance = distx;
      return;
    }

    // hitting Y faces?
    if (checkVaryingY<trdTypeT>(trd)) {
      if (FaceTrajectoryIntersection<Real_v, true, false, false>(trd, point, dir, disty)) {
        distance = disty;
        return;
      }

      if (FaceTrajectoryIntersection<Real_v, true, true, false>(trd, point, dir, disty)) distance = disty;
    } else {
      Real_v plane = trd.fDY1;
      if (dir.y() < Real_v(0.)) plane = Real_v(-trd.fDY1);
      disty       = (plane - point.y()) / dir.y();
      Real_v zhit = point.z() + disty * dir.z();
      Real_v xhit = point.x() + disty * dir.x();
      Real_v dx   = trd.fHalfX1plusX2 - trd.fFx * zhit;
      bool oky    = vecCore::math::Abs(xhit) < MakePlusTolerant<true>(dx) &&
                 vecCore::math::Abs(zhit) < MakePlusTolerant<true>(trd.fDZ);
      if (oky) distance = disty;
    }
    if (vecCore::math::Abs(distance) < kHalfTolerance) distance = Real_v(0.0);
  }

  /// @brief Computes safety from an outside point to the Trd boundary.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Query point in the Trd local frame.
  /// @param safety Output distance estimate to enter the Trd.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &trd,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace TrdUtilities;
    Safety<Real_v, trdTypeT, false>(trd, point, safety);
  }

  /// @brief Computes safety from an inside point to the Trd boundary.
  /// @tparam Real_v Scalar or vector floating-point type.
  /// @param trd Trd data structure with cached geometric coefficients.
  /// @param point Query point in the Trd local frame.
  /// @param safety Output distance estimate to exit the Trd.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &trd,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace TrdUtilities;
    Safety<Real_v, trdTypeT, true>(trd, point, safety);
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
