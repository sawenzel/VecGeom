/*
 * CutTubeImplementation.h
 *
 *  Created on: 03.11.2016
 *      Author: mgheata
 */

#ifndef VECGEOM_VOLUMES_KERNEL_CUTTUBEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_CUTTUBEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/CutTubeStruct.h"
#include "TubeImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct CutTubeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, CutTubeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCutTube;
class UnplacedCutTube;

template <typename T>
struct CutTubeStruct;

struct CutTubeImplementation {

  using PlacedShape_t    = PlacedCutTube;
  using UnplacedStruct_t = CutTubeStruct<Precision>;
  using UnplacedVolume_t = UnplacedCutTube;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, bool &inside);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToInKernel(UnplacedStruct_t const &unplaced,
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
                                                                        Vector3D<Real_v> &normal, bool &valid);
}; // End struct CutTubeImplementation

//********************************
//**** implementations start here
//********************************/

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::Contains(UnplacedStruct_t const &unplaced,
                                                                                  Vector3D<Real_v> const &point,
                                                                                  bool &contains)
{
  contains = false;

  bool inside_cutplanes = false;
  unplaced.GetCutPlanes().Contains<Real_v>(point, inside_cutplanes);

  if (!inside_cutplanes) return;

  TubeImplementation<TubeTypes::UniversalTube>::Contains<Real_v>(unplaced.GetTubeStruct(), point, contains);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::Inside(UnplacedStruct_t const &unplaced,
                                                                                Vector3D<Real_v> const &point,
                                                                                Inside_t &inside)
{
  Inside_t inside_cutplanes = EInside::kOutside;
  unplaced.GetCutPlanes().Inside<Real_v>(point, inside_cutplanes);

  if (inside_cutplanes == EInside::kOutside) {
    inside = inside_cutplanes;
    return;
  }

  TubeImplementation<TubeTypes::UniversalTube>::Inside<Real_v>(unplaced.GetTubeStruct(), point, inside);

  if (inside_cutplanes == EInside::kSurface && inside != EInside::kOutside) inside = inside_cutplanes;
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                                      Vector3D<Real_v> const &pointT,
                                                                                      Vector3D<Real_v> const &dir,
                                                                                      Real_v const &stepMax,
                                                                                      Real_v &distance)
{
  Vector3D<Real_v> point = pointT;
  const Real_v ptDist    = point.Mag();
  Real_v distToMove(0.);
  const Precision order = 100.;

  // Move very distant points closer before calling the hot kernel to avoid
  // unnecessary work on huge coordinates.
  if (ptDist > order * unplaced.fMaxVal) {
    distToMove = ptDist - Real_v(order * unplaced.fMaxVal);
    point += distToMove * dir;
  }

  DistanceToInKernel<Real_v>(unplaced, point, dir, stepMax, distance);
  distance += distToMove;
}
//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::DistanceToInKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{
  Vector3D<Real_v> propagated = point;
  distance                    = InfinityLength<Real_v>();

  auto const &cutplanes = unplaced.GetCutPlanes();
  auto const &plane0    = cutplanes.GetCutPlane(0);
  auto const &plane1    = cutplanes.GetCutPlane(1);
  const Real_v pdist0   = plane0.DistPlane(point);
  const Real_v pdist1   = plane1.DistPlane(point);
  const Real_v ndd0     = direction.Dot(Vector3D<Real_v>(plane0.GetNormal()));
  const Real_v ndd1     = direction.Dot(Vector3D<Real_v>(plane1.GetNormal()));

  Inside_t cutplane_state = (pdist0 < Real_v(0.0) && pdist1 < Real_v(0.0)) ? EInside::kInside : EInside::kOutside;
  if (vecCore::math::Abs(pdist0) < Real_v(kTolerance) || vecCore::math::Abs(pdist1) < Real_v(kTolerance)) {
    cutplane_state = EInside::kSurface;
  }

  Inside_t tube_state = EInside::kOutside;
  TubeImplementation<TubeTypes::UniversalTube>::Inside<Real_v>(unplaced.GetTubeStruct(), point, tube_state);

  if (cutplane_state == EInside::kOutside) {
    tube_state = cutplane_state;
  } else if (cutplane_state == EInside::kSurface && tube_state != EInside::kOutside) {
    if ((vecCore::math::Abs(pdist0) < Real_v(kTolerance) && pdist1 < Real_v(kTolerance) && ndd0 < Real_v(0.)) ||
        (vecCore::math::Abs(pdist1) < Real_v(kTolerance) && pdist0 < Real_v(kTolerance) && ndd1 < Real_v(0.))) {
      distance = Real_v(0.);
      return;
    }
    tube_state = cutplane_state;
  }

  if (tube_state == EInside::kInside) {
    distance = Real_v(-1.);
    return;
  }

  Real_v dplanes = Real_v(0.);
  if (cutplane_state != EInside::kInside) {
    Real_v dplane0 = -InfinityLength<Real_v>();
    if (ndd0 < Real_v(0.) && pdist0 > Real_v(-kTolerance)) dplane0 = -pdist0 / NonZero(ndd0);
    Real_v dplane1 = -InfinityLength<Real_v>();
    if (ndd1 < Real_v(0.) && pdist1 > Real_v(-kTolerance)) dplane1 = -pdist1 / NonZero(ndd1);
    dplanes              = vecCore::math::Max(dplane0, dplane1);
    const bool hitplanes = vecCore::math::Abs(dplanes) < stepMax && dplanes > Real_v(-kTolerance);
    if (!hitplanes) return;

    propagated += dplanes * direction;
    // Hitting a cut plane does not guarantee that the propagated point is
    // already between the two cut planes.
    cutplanes.Inside<Real_v>(propagated, cutplane_state);
    if (cutplane_state == EInside::kOutside) return;

    TubeImplementation<TubeTypes::UniversalTube>::Inside<Real_v>(unplaced.GetTubeStruct(), propagated, tube_state);
    if (tube_state != EInside::kOutside) {
      distance = vecCore::math::Abs(dplanes) < Real_v(kTolerance) ? Real_v(0.) : dplanes;
      return;
    }
  }

  Real_v dexit = InfinityLength<Real_v>();
  cutplanes.DistanceToOut<Real_v>(propagated, direction, dexit);

  Real_v dtube = InfinityLength<Real_v>();
  TubeImplementation<TubeTypes::UniversalTube>::DistanceToInKernel<Real_v>(unplaced.GetTubeStruct(), propagated,
                                                                           direction, stepMax, dtube);
  // Propagation to the cut planes can put the point inside the tube, so
  // DistanceToIn may return -1. Treat that as an immediate tube hit.
  if (dtube < Real_v(0.)) dtube = Real_v(0.);
  if (dexit < dtube) return;

  const Real_v candidate = dtube + dplanes;
  if (candidate < stepMax) distance = candidate;
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::DistanceToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{
  auto const &cutplanes = unplaced.GetCutPlanes();
  auto const &plane0    = cutplanes.GetCutPlane(0);
  auto const &plane1    = cutplanes.GetCutPlane(1);
  const Real_v pdist0   = plane0.DistPlane(point);
  const Real_v pdist1   = plane1.DistPlane(point);
  const Real_v ndd0     = direction.Dot(Vector3D<Real_v>(plane0.GetNormal()));
  const Real_v ndd1     = direction.Dot(Vector3D<Real_v>(plane1.GetNormal()));

  Real_v dist0 = InfinityLength<Real_v>();
  if (pdist0 > Real_v(kTolerance)) {
    dist0 = -InfinityLength<Real_v>();
  } else if (ndd0 > Real_v(0.) && pdist0 < Real_v(kTolerance)) {
    dist0 = -pdist0 / NonZero(ndd0);
  }

  Real_v dist1 = InfinityLength<Real_v>();
  if (pdist1 > Real_v(kTolerance)) {
    dist1 = -InfinityLength<Real_v>();
  } else if (ndd1 > Real_v(0.) && pdist1 < Real_v(kTolerance)) {
    dist1 = -pdist1 / NonZero(ndd1);
  }

  distance = vecCore::math::Min(dist0, dist1);
  if (distance < Real_v(0.)) {
    if ((vecCore::math::Abs(pdist0) < Real_v(kTolerance) && pdist1 < Real_v(kTolerance) &&
         ndd0 > -kToleranceDist<Real_v>) ||
        (vecCore::math::Abs(pdist1) < Real_v(kTolerance) && pdist0 < Real_v(kTolerance) &&
         ndd1 > -kToleranceDist<Real_v>)) {
      distance = Real_v(0.);
    }
  }

  Real_v dtube = InfinityLength<Real_v>();
  TubeImplementation<TubeTypes::UniversalTube>::DistanceToOut<Real_v>(unplaced.GetTubeStruct(), point, direction,
                                                                      stepMax, dtube);
  if (dtube < distance) distance = dtube;
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    Real_v &safety)
{
  unplaced.GetCutPlanes().SafetyToIn<Real_v>(point, safety);

  Real_v saftube;
  TubeImplementation<TubeTypes::UniversalTube>::SafetyToIn<Real_v>(unplaced.GetTubeStruct(), point, saftube);

  if (saftube > safety) safety = saftube;
}

//______________________________________________________________________________
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                Vector3D<Real_v> const &point, Real_v &safety)
{
  unplaced.GetCutPlanes().SafetyToOut<Real_v>(point, safety);

  Real_v saftube;
  TubeImplementation<TubeTypes::UniversalTube>::SafetyToOut<Real_v>(unplaced.GetTubeStruct(), point, saftube);

  if (saftube < safety) safety = saftube;
}

//______________________________________________________________________________
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void CutTubeImplementation::NormalKernel(UnplacedStruct_t const &unplaced,
                                                                 Vector3D<Real_v> const &point,
                                                                 Vector3D<Real_v> &normal, bool &valid)
{
  valid = true;

  Real_v safcut;
  unplaced.GetCutPlanes().SafetyToOut<Real_v>(point, safcut);

  Real_v saftube;
  TubeImplementation<TubeTypes::UniversalTube>::SafetyToOut<Real_v>(unplaced.GetTubeStruct(), point, saftube);

  if (vecCore::math::Abs(saftube) < vecCore::math::Abs(safcut)) {
    TubeImplementation<TubeTypes::UniversalTube>::NormalKernel<Real_v>(unplaced.GetTubeStruct(), point, normal, valid);
    return;
  }

  normal = unplaced.GetCutPlanes().GetNormal(point.z() < 0 ? 0 : 1);
}

//*****************************
//**** Implementations end here
//*****************************
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_KERNEL_CUTTUBEIMPLEMENTATION_H_ */
