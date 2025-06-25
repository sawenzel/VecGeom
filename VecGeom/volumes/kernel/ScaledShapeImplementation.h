/// @file ScaledShapeImplementation.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SCALEDSHAPEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SCALEDSHAPEIMPLEMENTATION_H_

#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/ScaledShapeStruct.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ScaledShapeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ScaledShapeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedScaledShape;
class UnplacedScaledShape;

// template <typename T>
// struct ScaledShapeStruct;

struct ScaledShapeImplementation {

  using PlacedShape_t    = PlacedScaledShape;
  using UnplacedStruct_t = ScaledShapeStruct<Precision>;
  using UnplacedVolume_t = UnplacedScaledShape;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside);

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &normal,
                                                                        vecCore::Mask_v<Real_v> &valid);

}; // End struct ScaledShapeImplementation

// Implementations start here

template <typename Real_v, typename Bool_v>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::Contains(UnplacedStruct_t const &unplaced,
                                                                 Vector3D<Real_v> const &point, Bool_v &inside)
{

  // Transform local point to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Now call Contains for the unscaled shape
  inside = unplaced.fPlaced->Contains(ulocalPoint);
}

template <typename Real_v, typename Inside_t>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::Inside(UnplacedStruct_t const &unplaced,
                                                               Vector3D<Real_v> const &point, Inside_t &inside)
{

  // Transform local point to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Now call Inside for the unscaled shape
  inside = unplaced.fPlaced->Inside(ulocalPoint);
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                     Vector3D<Real_v> const &point,
                                                                     Vector3D<Real_v> const &direction,
                                                                     Real_v const &stepMax, Real_v &distance)
{

  // Transform point, direction and stepMax to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Direction is un-normalized after scale transformation
  Vector3D<Real_v> ulocalDir;
  unplaced.fScale.Transform(direction, ulocalDir);
  ulocalDir.Normalize();

  auto ustepMax = unplaced.fScale.TransformDistance(stepMax, direction);

  // Compute distance in unscaled system
  distance = unplaced.fPlaced->DistanceToIn(ulocalPoint, ulocalDir, ustepMax);

  // Convert distance back to master (leave unchanged if it was infinity)
  vecCore__MaskedAssignFunc(distance, distance < InfinityLength<Real_v>(),
                            unplaced.fScale.InverseTransformDistance(distance, ulocalDir));
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point,
                                                                      Vector3D<Real_v> const &direction,
                                                                      Real_v const &stepMax, Real_v &distance)
{

  // Transform point, direction and stepMax to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Direction is un-normalized after scale transformation
  Vector3D<Real_v> ulocalDir;
  unplaced.fScale.Transform(direction, ulocalDir);
  ulocalDir.Normalize();

  auto ustepMax = unplaced.fScale.TransformDistance(stepMax, direction);

  // Compute distance in unscaled system
  distance = unplaced.fPlaced->DistanceToOut(ulocalPoint, ulocalDir, ustepMax);

  // Convert distance back to master (leave unchanged if it was infinity)
  vecCore__MaskedAssignFunc(distance, distance < InfinityLength<Real_v>(),
                            unplaced.fScale.InverseTransformDistance(distance, ulocalDir));
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                   Vector3D<Real_v> const &point, Real_v &safety)
{

  // Transform point to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Compute unscaled safety, then scale it.
  safety = unplaced.fPlaced->SafetyToIn(ulocalPoint);
  safety = unplaced.fScale.InverseTransformSafety(safety);
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Real_v &safety)
{

  // Transform point to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Compute unscaled safety, then scale it.
  safety = unplaced.fPlaced->SafetyToOut(ulocalPoint);
  safety = unplaced.fScale.InverseTransformSafety(safety);
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void ScaledShapeImplementation::NormalKernel(UnplacedStruct_t const &unplaced,
                                                                     Vector3D<Real_v> const &point,
                                                                     Vector3D<Real_v> &normal,
                                                                     vecCore::Mask_v<Real_v> &valid)
{

  // Transform point to unscaled shape frame
  Vector3D<Real_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Compute normal in unscaled frame
  Vector3D<Real_v> ulocalNorm;
  valid = unplaced.fPlaced->Normal(ulocalPoint, ulocalNorm);

  // Convert normal to scaled frame
  unplaced.fScale.InverseTransformNormal(ulocalNorm, normal);
  normal.Normalize(); // <- Is this actually needed? PlacedVolume::Normal is supposed to return normalized normal
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_SCALEDSHAPEIMPLEMENTATION_H_
