// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file implements the algorithms for Parallelepiped
/// @file volumes/kernel/ParallelepipedImplementation.h
/// @author First version by Johannes de Fine Licht
/// @author Revised by Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/ParallelepipedStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ParallelepipedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ParallelepipedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedParallelepiped;
template <typename T>
struct ParallelepipedStruct;
class UnplacedParallelepiped;

struct ParallelepipedImplementation {

  using PlacedShape_t    = PlacedParallelepiped;
  using UnplacedStruct_t = ParallelepipedStruct<Precision>;
  using UnplacedVolume_t = UnplacedParallelepiped;

  /// Transform a point or direction to the oblique box frame.
  /// In this frame the parallelepiped is represented by an axis-aligned box
  /// with half lengths stored in `fDimensions`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void TransformToBoxFrame(UnplacedStruct_t const &unplaced,
                                                                               Vector3D<Real_v> &point)
  {
    point.y() -= unplaced.fTanThetaSinPhi * point.z();
    point.x() -= unplaced.fTanThetaCosPhi * point.z() + unplaced.fTanAlpha * point.y();
  }

  /// Transform a point or direction to scalar box-frame components.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void TransformToBoxFrame(UnplacedStruct_t const &unplaced,
                                                                               Vector3D<Real_v> const &point, Real_v &x,
                                                                               Real_v &y, Real_v &z)
  {
    z = point.z();
    y = point.y() - unplaced.fTanThetaSinPhi * z;
    x = point.x() - unplaced.fTanThetaCosPhi * z - unplaced.fTanAlpha * y;
  }

  /// Return signed distances to the three pairs of limiting planes.
  /// Positive components are outside, zero is on the corresponding surface.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void PlaneSafetyVector(UnplacedStruct_t const &unplaced,
                                                                             Vector3D<Real_v> const &localPoint,
                                                                             Vector3D<Real_v> &safety)
  {
    safety = localPoint.Abs() - Vector3D<Real_v>(unplaced.fDimensions);
    safety.x() *= unplaced.fCtx;
    safety.y() *= unplaced.fCty;
  }

  /// Return the largest signed distance to the limiting plane pairs.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v MaxPlaneSafety(UnplacedStruct_t const &unplaced,
                                                                            Real_v const &x, Real_v const &y,
                                                                            Real_v const &z)
  {
    const Real_v sx = (Abs(x) - unplaced.fDimensions.x()) * unplaced.fCtx;
    const Real_v sy = (Abs(y) - unplaced.fDimensions.y()) * unplaced.fCty;
    const Real_v sz = Abs(z) - unplaced.fDimensions.z();
    return Max(sx, sy, sz);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    Real_v x, y, z;
    TransformToBoxFrame<Real_v>(unplaced, point, x, y, z);

    inside = MaxPlaneSafety(unplaced, x, y, z) < Real_v(kHalfTolerance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    Real_v x, y, z;
    TransformToBoxFrame<Real_v>(unplaced, point, x, y, z);

    const Real_v safety = MaxPlaneSafety(unplaced, x, y, z);
    inside = Abs(safety) < Real_v(kHalfTolerance) ? Inside_t(kSurface)
                                                  : (safety < Real_v(0.0) ? Inside_t(kInside) : Inside_t(kOutside));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v x, y, z;
    TransformToBoxFrame<Real_v>(unplaced, point, x, y, z);

    safety = MaxPlaneSafety(unplaced, x, y, z);
    safety = Abs(safety) < Real_v(kHalfTolerance) ? Real_v(0.0) : safety;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    Real_v x, y, z;
    TransformToBoxFrame<Real_v>(unplaced, point, x, y, z);

    safety = -MaxPlaneSafety(unplaced, x, y, z);
    safety = Abs(safety) < Real_v(kHalfTolerance) ? Real_v(0.0) : safety;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    // Transform point and direction to local (oblique) system of coordinates,
    // compute safety vector
    Vector3D<Real_v> p(point);
    Vector3D<Real_v> v(direction);

    TransformToBoxFrame<Real_v>(unplaced, p);
    TransformToBoxFrame<Real_v>(unplaced, v);

    BoxImplementation::DistanceToIn(BoxStruct<Precision>(unplaced.fDimensions), p, v, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /*stepMax*/, Real_v &distance)
  {
    Real_v px, py, pz;
    Real_v vx, vy, vz;
    TransformToBoxFrame<Real_v>(unplaced, point, px, py, pz);
    TransformToBoxFrame<Real_v>(unplaced, direction, vx, vy, vz);

    distance            = Real_v(-1.);
    const Real_v dx     = unplaced.fDimensions.x();
    const Real_v dy     = unplaced.fDimensions.y();
    const Real_v dz     = unplaced.fDimensions.z();
    const Real_v tol    = kToleranceDist<Real_v>;
    const Real_v safety = Max((Abs(px) - dx) * unplaced.fCtx, (Abs(py) - dy) * unplaced.fCty, Abs(pz) - dz);
    if (safety > tol) return;

    distance              = InfinityLength<Real_v>();
    const Real_v absVx    = Abs(vx);
    const Real_v absVy    = Abs(vy);
    const Real_v absVz    = Abs(vz);
    const Real_v zeroDist = Real_v(0.);

    if (absVx * dx >= tol) distance = Min(distance, (dx - Sign(vx) * px) * (Real_v(1.) / absVx));
    if (absVy * dy >= tol) distance = Min(distance, (dy - Sign(vy) * py) * (Real_v(1.) / absVy));
    if (absVz * dz >= tol) distance = Min(distance, (dz - Sign(vz) * pz) * (Real_v(1.) / absVz));

    if (distance < zeroDist) distance = zeroDist;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {
    // Compute normal at the point on the surface.
    // In case the point is not on the surface, set valid = false.
    // Must return a valid vector (even if the point is not on the surface).
    // On edge or corner, provide an average normal of all facets within the tolerance.
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v px, py, pz;
    TransformToBoxFrame<Real_v>(unplaced, point, px, py, pz);
    const Real_v safetyX = (Abs(px) - unplaced.fDimensions.x()) * unplaced.fCtx;
    const Real_v safetyY = (Abs(py) - unplaced.fDimensions.y()) * unplaced.fCty;
    const Real_v safetyZ = Abs(pz) - unplaced.fDimensions.z();

    // Set normal
    const Vector3D<Real_v> signs(Sign(px), Sign(py), Sign(pz));
    if (Abs(safetyZ) <= kHalfTolerance) normal = Vector3D<Real_v>(0., 0., signs.z());
    if (Abs(safetyY) <= kHalfTolerance) normal = normal + signs.y() * unplaced.fNormals[1];
    if (Abs(safetyX) <= kHalfTolerance) normal = normal + signs.x() * unplaced.fNormals[0];

    Real_v mag2 = normal.Mag2();
    if (mag2 > 1.) normal = normal.Unit();
    if (mag2 > Real_v(0.)) return normal;

    // Point is not on the surface - normally, this should never be.
    // Return normal of the nearest face.
    valid         = false;
    Real_v safety = Max(safetyX, safetyY, safetyZ);
    normal        = signs.x() * unplaced.fNormals[0];
    if (safetyY == safety) normal = signs.y() * unplaced.fNormals[1];
    if (safetyZ == safety) normal = signs.z() * unplaced.fNormals[2];
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
