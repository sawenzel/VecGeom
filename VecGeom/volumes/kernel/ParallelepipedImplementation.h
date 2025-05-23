// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file implements the algorithms for Paralleliped
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

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Transform(UnplacedStruct_t const &unplaced,
                                                                     Vector3D<Real_v> &point)
  {
    point.y() -= unplaced.fTanThetaSinPhi * point.z();
    point.x() -= unplaced.fTanThetaCosPhi * point.z() + unplaced.fTanAlpha * point.y();
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyVector(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &localPoint,
                                                                        Vector3D<Real_v> &safety)
  {
    safety = localPoint.Abs() - Vector3D<Real_v>(unplaced.fDimensions);
    safety.x() *= unplaced.fCtx;
    safety.y() *= unplaced.fCty;
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Vector3D<Real_v> localPoint(point);
    Vector3D<Real_v> safetyVector;
    Transform<Real_v>(unplaced, localPoint);
    SafetyVector<Real_v>(unplaced, localPoint, safetyVector);

    inside = safetyVector.Max() < Real_v(0.0);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    Vector3D<Real_v> localPoint(point);
    Vector3D<Real_v> safetyVector;
    Transform<Real_v>(unplaced, localPoint);
    SafetyVector<Real_v>(unplaced, localPoint, safetyVector);

    Real_v safety = safetyVector.Max();
    inside        = vecCore::Blend(safety < Real_v(0.0), Inside_v(kInside), Inside_v(kOutside));
    vecCore__MaskedAssignFunc(inside, vecCore::math::Abs(safety) < Real_v(kHalfTolerance), Inside_v(kSurface));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    Vector3D<Real_v> localPoint(point);
    Vector3D<Real_v> safetyVector;
    Transform<Real_v>(unplaced, localPoint);
    SafetyVector<Real_v>(unplaced, localPoint, safetyVector);

    safety = safetyVector.Max();
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) < Real_v(kHalfTolerance), Real_v(0.0));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    Vector3D<Real_v> localPoint(point);
    Vector3D<Real_v> safetyVector;
    Transform<Real_v>(unplaced, localPoint);
    SafetyVector<Real_v>(unplaced, localPoint, safetyVector);

    safety = -safetyVector.Max();
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) < Real_v(kHalfTolerance), Real_v(0.0));
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

    Transform<Real_v>(unplaced, p);
    Transform<Real_v>(unplaced, v);

    BoxImplementation::DistanceToIn(BoxStruct<Precision>(unplaced.fDimensions), p, v, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    // Transform point and direction to local (oblique) system of coordinates,
    // compute safety vector
    Vector3D<Real_v> p(point);
    Vector3D<Real_v> v(direction);

    Transform<Real_v>(unplaced, p);
    Transform<Real_v>(unplaced, v);

    BoxImplementation::DistanceToOut(BoxStruct<Precision>(unplaced.fDimensions), p, v, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // Compute normal at the point on the surface.
    // In case the point is not on the surface, set valid = false.
    // Must return a valid vector (even if the point is not on the surface).
    // On edge or corner, provide an average normal of all facets within the tolerance.
    Vector3D<Real_v> normal(0.);
    valid = true;

    // Transform point to local (oblique) system of coordinates and compute safety vector
    Vector3D<Real_v> p(point);
    Vector3D<Real_v> safetyVector;
    Transform<Real_v>(unplaced, p);
    SafetyVector<Real_v>(unplaced, p, safetyVector);

    // Set normal
    const Vector3D<Real_v> signs(Sign(p.x()), Sign(p.y()), Sign(p.z()));
    vecCore__MaskedAssignFunc(normal, Abs(safetyVector.z()) <= kHalfTolerance, Vector3D<Real_v>(0., 0., signs.z()));
    vecCore__MaskedAssignFunc(normal, Abs(safetyVector.y()) <= kHalfTolerance,
                              normal + signs.y() * unplaced.fNormals[1]);
    vecCore__MaskedAssignFunc(normal, Abs(safetyVector.x()) <= kHalfTolerance,
                              normal + signs.x() * unplaced.fNormals[0]);

    Real_v mag2 = normal.Mag2();
    vecCore__MaskedAssignFunc(normal, mag2 > 1., normal.Unit());
    if (vecCore::MaskFull(mag2 > Real_v(0.))) return normal;

    // Point is not on the surface - normally, this should never be.
    // Return normal of the nearest face.
    vecCore__MaskedAssignFunc(valid, mag2 == Real_v(0.), false);
    Real_v safety = safetyVector.Max();
    normal        = signs.x() * unplaced.fNormals[0];
    vecCore__MaskedAssignFunc(normal, safetyVector.y() == safety, signs.y() * unplaced.fNormals[1]);
    vecCore__MaskedAssignFunc(normal, safetyVector.z() == safety, signs.z() * unplaced.fNormals[2]);
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
