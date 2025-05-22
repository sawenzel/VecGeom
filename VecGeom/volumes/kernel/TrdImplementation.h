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

/*
 * Checks whether a point (x, y) falls on the left or right half-plane
 * of a line. The line is defined by a (vx, vy) vector, extended to infinity.
 *
 * Of course this can only be used for lines that pass through (0, 0), but
 * you can supply transformed coordinates for the point to check for any line.
 *
 * This simply calculates the magnitude of the cross product of vectors (px, py)
 * and (vx, vy), which is defined as |x| * |v| * sin theta.
 *
 * If the cross product is positive, the point is clockwise of V, or the "right"
 * half-plane. If it's negative, the point is CCW and on the "left" half-plane.
 */

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PointLineOrientation(Real_v const &px, Real_v const &py,
                                                                       Precision const &vx, Precision const &vy,
                                                                       Real_v &crossProduct)
{
  crossProduct = vx * py - vy * px;
}

/*
 * Check intersection of the trajectory of a particle with a segment
 * that's bound from -Ylimit to +Ylimit.j
 *
 * All points of the along-vector of a plane lie on
 * s * (alongX, alongY)
 * All points of the trajectory of the particle lie on
 * (x, y) + t * (vx, vy)
 * Thefore, it must hold that s * (alongX, alongY) == (x, y) + t * (vx, vy)
 * Solving by t we get t = (alongY*x - alongX*y) / (vy*alongX - vx*alongY)
 *
 * t gives the distance, but how to make sure hitpoint is inside the
 * segment and not just the infinite line defined by the segment?
 *
 * Check that |hity| <= Ylimit
 */

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PlaneTrajectoryIntersection(
    Real_v const &alongX, Real_v const &alongY, Real_v const &ylimit, Real_v const &posx, Real_v const &posy,
    Real_v const &dirx, Real_v const &diry, Real_v &dist, vecCore::Mask_v<Real_v> &ok)
{
  dist = (alongY * posx - alongX * posy) / (diry * alongX - dirx * alongY);

  Real_v hity = posy + dist * diry;
  ok          = vecCore::math::Abs(hity) <= ylimit && dist > 0;
}

template <typename Real_v, bool forY, bool mirroredPoint, bool toInside>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void FaceTrajectoryIntersection(TrdStruct<Precision> const &trd,
                                                                             Vector3D<Real_v> const &pos,
                                                                             Vector3D<Real_v> const &dir, Real_v &dist,
                                                                             vecCore::Mask_v<Real_v> &ok)
{
  Real_v alongV, posV, dirV, posK, dirK, fV, fK, halfKplus, v1;
  //    fNormals[0].Set(-fCalfX, 0., fFx*fCalfX);
  //    fNormals[1].Set(fCalfX, 0., fFx*fCalfX);
  //    fNormals[2].Set(0., -fCalfY, fFy*fCalfY);
  //    fNormals[3].Set(0., fCalfY, fFy*fCalfY);
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
  if (toInside)
    ok = ndotv_alongZ < -kTolerance;
  else
    ok = ndotv_alongZ > kTolerance;
  if (vecCore::MaskEmpty(ok)) return;

  // distance from trajectory to face
  dist = (alongZ * (posV - v1) - alongV * (pos.z() + trd.fDZ)) / (dir.z() * alongV - dirV * alongZ + kTiny);
  ok &= dist > Real_v(MakeMinusTolerant<true>(0.));
  if (!vecCore::MaskEmpty(ok)) {
    // need to make sure z hit falls within bounds
    Real_v hitz = pos.z() + dist * dir.z();
    ok &= vecCore::math::Abs(hitz) < MakePlusTolerant<true>(trd.fDZ);
    // need to make sure hit on varying dimension falls within bounds
    Real_v hitk = posK + dist * dirK;
    Real_v dK   = halfKplus - fK * hitz; // calculate the width of the varying dimension at hitz
    ok &= vecCore::math::Abs(hitk) < MakePlusTolerant<true>(dK);
    vecCore::MaskedAssign(dist, ok & (vecCore::math::Abs(dist) < kHalfTolerance), Real_v(0.0));
  }
}

template <typename Real_v, typename trdTypeT, bool inside>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Safety(TrdStruct<Precision> const &trd, Vector3D<Real_v> const &pos,
                                                         Real_v &dist)
{
  using namespace TrdTypes;
  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v safz = trd.fDZ - vecCore::math::Abs(pos.z());
  // std::cerr << "safz: " << safz << std::endl;
  dist = safz;

  Real_v distx = trd.fHalfX1plusX2 - trd.fFx * pos.z();
  Bool_v okx   = distx >= 0;
  Real_v safx  = (distx - vecCore::math::Abs(pos.x())) * trd.fCalfX;
  vecCore::MaskedAssign(dist, okx && safx < dist, safx);
  // std::cerr << "safx: " << safx << std::endl;

  if (checkVaryingY<trdTypeT>(trd)) {
    Real_v disty = trd.fHalfY1plusY2 - trd.fFy * pos.z();
    Bool_v oky   = disty >= 0;
    Real_v safy  = (disty - vecCore::math::Abs(pos.y())) * trd.fCalfY;
    vecCore::MaskedAssign(dist, oky && safy < dist, safy);
  } else {
    Real_v safy = trd.fDY1 - vecCore::math::Abs(pos.y());
    vecCore::MaskedAssign(dist, safy < dist, safy);
  }
  if (!inside) dist = -dist;
}

template <typename Real_v, typename trdTypeT, bool surfaceT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedInside(TrdStruct<Precision> const &trd,
                                                                        Vector3D<Real_v> const &point,
                                                                        vecCore::Mask_v<Real_v> &completelyinside,
                                                                        vecCore::Mask_v<Real_v> &completelyoutside)
{

  using namespace TrdUtilities;
  using namespace TrdTypes;

  Real_v pzPlusDz = point.z() + trd.fDZ;

  // inside Z?
  completelyoutside = vecCore::math::Abs(point.z()) > MakePlusTolerant<surfaceT>(trd.fDZ);
  if (surfaceT) completelyinside = vecCore::math::Abs(point.z()) < MakeMinusTolerant<surfaceT>(trd.fDZ);

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

template <typename trdTypeT>
struct TrdImplementation {

  using UnplacedStruct_t = TrdStruct<Precision>;
  using UnplacedVolume_t = SUnplacedTrd<trdTypeT>;
  using PlacedShape_t    = SPlacedTrd<UnplacedVolume_t>;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContains(UnplacedStruct_t const &trd,
                                                                            Vector3D<Real_v> const &point,
                                                                            Bool_v &inside)
  {

    Bool_v unused;
    TrdUtilities::UnplacedInside<Real_v, trdTypeT, false>(trd, point, unused, inside);
    inside = !inside;
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &trd,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {

    Bool_v unused;
    TrdUtilities::UnplacedInside<Real_v, trdTypeT, false>(trd, point, unused, inside);
    inside = !inside;
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &trd,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    // use double-based vector for result, as Bool_v is a mask for precision_v
    using Bool_v = vecCore::Mask_v<Real_v>;
    const Real_v in(EInside::kInside);
    const Real_v out(EInside::kOutside);
    Bool_v inmask  = Bool_v(false);
    Bool_v outmask = Bool_v(false);
    Real_v result(EInside::kSurface);

    TrdUtilities::UnplacedInside<Real_v, trdTypeT, true>(trd, point, inmask, outmask);

    vecCore::MaskedAssign(result, inmask, in);
    vecCore::MaskedAssign(result, outmask, out);

    // Manual conversion from double to int here is necessary because int_v and
    // precision_v have different number of elements in SIMD vector, so Bool_v
    // (mask for precision_v) cannot be cast to mask for inside, which is a
    // different type and does not exist in the current backend system
    for (size_t i = 0; i < vecCore::VectorSize(result); i++)
      vecCore::Set(inside, i, vecCore::Get(result, i));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &trd,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {

    using namespace TrdUtilities;
    using namespace TrdTypes;
    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v hitx, hity;
    // Real_v hitz;

    Vector3D<Real_v> pos_local;
    Vector3D<Real_v> dir_local;
    distance = InfinityLength<Real_v>();

    // hit Z faces?
    Bool_v inz   = vecCore::math::Abs(point.z()) < Real_v(MakeMinusTolerant<true>(trd.fDZ));
    Real_v distx = trd.fHalfX1plusX2 - trd.fFx * point.z();
    Bool_v inx   = (distx - vecCore::math::Abs(point.x())) * trd.fCalfX > Real_v(MakePlusTolerant<true>(0.));
    Real_v disty;
    Bool_v iny;
    if (checkVaryingY<trdTypeT>(trd)) {
      disty = trd.fHalfY1plusY2 - trd.fFy * point.z();
      iny   = (disty - vecCore::math::Abs(point.y())) * trd.fCalfY > Real_v(MakePlusTolerant<true>(0.));
    } else {
      disty = vecCore::math::Abs(point.y()) - trd.fDY1;
      iny   = disty < Real_v(MakeMinusTolerant<true>(0.));
    }
    Bool_v inside = inx & iny & inz;
    vecCore__MaskedAssignFunc(distance, inside, Real_v(-1.));
    Bool_v done = inside;
    Bool_v okz  = point.z() * direction.z() < Real_v(0.);
    okz &= !inz;
    if (!vecCore::MaskEmpty(okz)) {
      Real_v distz = (vecCore::math::Abs(point.z()) - trd.fDZ) / vecCore::math::Abs(direction.z());
      // exclude case in which particle is going away
      hitx = vecCore::math::Abs(point.x() + distz * direction.x());
      hity = vecCore::math::Abs(point.y() + distz * direction.y());

      // hitting top face?
      Bool_v okzt = point.z() > (trd.fDZ - kHalfTolerance) && hitx <= trd.fDX2 && hity <= trd.fDY2;
      // hitting bottom face?
      Bool_v okzb = point.z() < (-trd.fDZ + kHalfTolerance) && hitx <= trd.fDX1 && hity <= trd.fDY1;

      okz &= (okzt | okzb);
      vecCore::MaskedAssign(distance, okz, distz);
    }
    done |= okz;
    if (vecCore::MaskFull(done)) {
      vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
      return;
    }

    // hitting X faces?
    Bool_v okx = Bool_v(false);
    if (!vecCore::MaskFull(inx)) {

      FaceTrajectoryIntersection<Real_v, false, false, true>(trd, point, direction, distx, okx);
      vecCore::MaskedAssign(distance, okx, distx);

      FaceTrajectoryIntersection<Real_v, false, true, true>(trd, point, direction, distx, okx);
      vecCore::MaskedAssign(distance, okx, distx);
    }
    done |= okx;
    if (vecCore::MaskFull(done)) {
      vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
      return;
    }

    // hitting Y faces?
    Bool_v oky;
    if (checkVaryingY<trdTypeT>(trd)) {
      if (!vecCore::MaskFull(iny)) {
        FaceTrajectoryIntersection<Real_v, true, false, true>(trd, point, direction, disty, oky);
        vecCore::MaskedAssign(distance, oky, disty);

        FaceTrajectoryIntersection<Real_v, true, true, true>(trd, point, direction, disty, oky);
        vecCore::MaskedAssign(distance, oky, disty);
      }
    } else {
      if (!vecCore::MaskFull(iny)) {
        disty /= vecCore::math::Abs(direction.y());
        Real_v zhit = point.z() + disty * direction.z();
        Real_v xhit = point.x() + disty * direction.x();
        Real_v dx   = trd.fHalfX1plusX2 - trd.fFx * zhit;
        oky         = point.y() * direction.y() < 0 && disty > -kHalfTolerance && vecCore::math::Abs(xhit) < dx &&
              vecCore::math::Abs(zhit) < trd.fDZ;
        vecCore::MaskedAssign(distance, oky, disty);
      }
    }
    vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &trd,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const & /*stepMax*/, Real_v &distance)
  {

    using namespace TrdUtilities;
    using namespace TrdTypes;
    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v hitx, hity;
    // Real_v hitz;
    distance = Real_v(0.0);

    // hit top Z face?
    Real_v invdir = Real_v(1.) / vecCore::math::Abs(dir.z() + kTiny);
    Real_v safz   = trd.fDZ - vecCore::math::Abs(point.z());
    Bool_v out    = safz < Real_v(MakeMinusTolerant<true>(0.));
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
    if (vecCore::MaskFull(out)) {
      distance = Real_v(-1.);
      return;
    }
    auto maxXY  = Max(trd.fHalfX1plusX2, trd.fHalfY1plusY2);
    Bool_v okzt = dir.z() * maxXY > kTolerance;
    if (!vecCore::MaskEmpty(okzt)) {
      Real_v distz = (trd.fDZ - point.z()) * invdir;
      hitx         = vecCore::math::Abs(point.x() + distz * dir.x());
      hity         = vecCore::math::Abs(point.y() + distz * dir.y());
      okzt &= hitx < MakePlusTolerant<true>(trd.fDX2) && hity < MakePlusTolerant<true>(trd.fDY2);
      vecCore::MaskedAssign(distance, okzt, distz);
      if (vecCore::MaskFull(okzt)) {
        vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
        return;
      }
    }

    // hit bottom Z face?
    Bool_v okzb = dir.z() * maxXY < -kTolerance;
    if (!vecCore::MaskEmpty(okzb)) {
      Real_v distz = (point.z() + trd.fDZ) * invdir;
      hitx         = vecCore::math::Abs(point.x() + distz * dir.x());
      hity         = vecCore::math::Abs(point.y() + distz * dir.y());
      okzb &= hitx < MakePlusTolerant<true>(trd.fDX1) && hity < MakePlusTolerant<true>(trd.fDY1);
      vecCore::MaskedAssign(distance, okzb, distz);
      if (vecCore::MaskFull(okzb)) {
        vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
        return;
      }
    }

    // hitting X faces?
    Bool_v okx;

    FaceTrajectoryIntersection<Real_v, false, false, false>(trd, point, dir, distx, okx);

    vecCore::MaskedAssign(distance, okx, distx);
    if (vecCore::MaskFull(okx)) {
      vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
      return;
    }

    FaceTrajectoryIntersection<Real_v, false, true, false>(trd, point, dir, distx, okx);
    vecCore::MaskedAssign(distance, okx, distx);
    if (vecCore::MaskFull(okx)) {
      vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
      return;
    }

    // hitting Y faces?
    Bool_v oky;

    if (checkVaryingY<trdTypeT>(trd)) {
      FaceTrajectoryIntersection<Real_v, true, false, false>(trd, point, dir, disty, oky);
      vecCore::MaskedAssign(distance, oky, disty);
      if (vecCore::MaskFull(oky)) {
        vecCore::MaskedAssign(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
        return;
      }

      FaceTrajectoryIntersection<Real_v, true, true, false>(trd, point, dir, disty, oky);
      vecCore::MaskedAssign(distance, oky, disty);
    } else {
      Real_v plane = trd.fDY1;
      vecCore__MaskedAssignFunc(plane, dir.y() < Real_v(0.), Real_v(-trd.fDY1));
      disty       = (plane - point.y()) / dir.y();
      Real_v zhit = point.z() + disty * dir.z();
      Real_v xhit = point.x() + disty * dir.x();
      Real_v dx   = trd.fHalfX1plusX2 - trd.fFx * zhit;
      oky         = vecCore::math::Abs(xhit) < MakePlusTolerant<true>(dx) &&
            vecCore::math::Abs(zhit) < MakePlusTolerant<true>(trd.fDZ);
      vecCore::MaskedAssign(distance, oky, disty);
    }
    vecCore__MaskedAssignFunc(distance, vecCore::math::Abs(distance) < kHalfTolerance, Real_v(0.0));
    vecCore__MaskedAssignFunc(distance, out, Real_v(-1.0));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &trd,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace TrdUtilities;
    Safety<Real_v, trdTypeT, false>(trd, point, safety);
  }

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
