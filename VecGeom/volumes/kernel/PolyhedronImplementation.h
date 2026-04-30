/// \file PolyhedronImplementation.h
/// \brief Polyhedron kernel helpers and public implementation entry points.
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include <cstdio>

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/TubeImplementation.h"
#include "VecGeom/volumes/Quadrilaterals.h"
#include "VecGeom/volumes/PolyhedronStruct.h"

namespace vecgeom {

// VECGEOM_DEVICE_FORWARD_DECLARE(struct PolyhedronImplementation;);

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(struct, PolyhedronImplementation, Polyhedron::EInnerRadii,
                                        Polyhedron::EInnerRadii::kGeneric, Polyhedron::EPhiCutout,
                                        Polyhedron::EPhiCutout::kGeneric);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolyhedron;
class UnplacedPolyhedron;

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
struct PolyhedronImplementation {

  using PlacedShape_t    = PlacedPolyhedron;
  using UnplacedStruct_t = PolyhedronStruct<Precision>;
  using UnplacedVolume_t = UnplacedPolyhedron;

  /// @brief Locate the z-segment owning a point.
  /// @param unplaced Polyhedron storage.
  /// @param pointZ Z-coordinate of the query point.
  /// @return Index of the owning z-segment. Returns `-1` below the first
  ///         z-plane and `N` above the last z-plane, where `N` is the segment
  ///         count.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static int FindZSegment(UnplacedStruct_t const &unplaced,
                                                                       Real_v const &pointZ);

  /// @brief Locate the phi-segment owning a point.
  /// @param unplaced Polyhedron storage.
  /// @param point Query point.
  /// @return Index of the owning phi-segment, or `-1` if no segment matches.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static int FindPhiSegment(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point);

  /// @brief Compute the first valid entry distance against one z-segment shell.
  /// @param unplaced Polyhedron storage.
  /// @param segmentIndex Z-segment index to test.
  /// @param point Ray origin.
  /// @param direction Ray direction.
  /// @return Distance to the closest accepted quadrilateral hit.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v DistanceToInZSegment(UnplacedStruct_t const &unplaced,
                                                                                  int segmentIndex,
                                                                                  Vector3D<Real_v> const &point,
                                                                                  Vector3D<Real_v> const &direction);

  /// @brief Compute the first valid exit distance against one z-segment shell.
  /// @param unplaced Polyhedron storage.
  /// @param segmentIndex Z-segment index to test.
  /// @param zMin Lower z bound of the owning segment.
  /// @param zMax Upper z bound of the owning segment.
  /// @param point Ray origin.
  /// @param direction Ray direction.
  /// @return Distance to the closest accepted quadrilateral hit.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v DistanceToOutZSegment(UnplacedStruct_t const &unplaced,
                                                                                   int segmentIndex, Precision zMin,
                                                                                   Precision zMax,
                                                                                   Vector3D<Real_v> const &point,
                                                                                   Vector3D<Real_v> const &direction);

  /// @brief Compute the exact squared safety to one z-segment shell.
  /// @param unplaced Polyhedron storage.
  /// @param segmentIndex Z-segment index to test.
  /// @param phiIndex Input/output phi-segment index, updated for the winning
  ///        feature.
  /// @param point Query point.
  /// @param pt_inside Whether the public caller classified the point as inside.
  /// @param iSurf Output identifier of the closest feature family.
  /// @return Exact squared distance to the closest quadrilateral feature of the
  ///         selected z-segment.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision SafetyToZSegmentSquared(UnplacedStruct_t const &unplaced, int segmentIndex, int &phiIndex,
                                           Vector3D<Precision> const &point, bool pt_inside, int &iSurf);

  /// @brief Minimize a distance-to-boundary candidate with the endcap distance.
  /// @param unplaced Polyhedron storage.
  /// @param goingRight Whether the ray advances towards increasing z.
  /// @param point Ray origin.
  /// @param direction Ray direction.
  /// @param distance Input/output minimum distance.
  template <bool pointInsideT>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToEndcaps(UnplacedStruct_t const &unplaced,
                                                                             bool goingRight,
                                                                             Vector3D<Precision> const &point,
                                                                             Vector3D<Precision> const &direction,
                                                                             Precision &distance);

  /// @brief Minimize a squared safety candidate with the closest endcap.
  /// @param unplaced Polyhedron storage.
  /// @param point Query point.
  /// @param distance Input/output squared safety estimate.
  /// @param iz Output normal direction for the winning endcap.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static void SafetyToEndcapsSquared(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                     Precision &distance, int &iz);

  /// @brief Check whether a point lies inside the infinite phi cutout wedge.
  /// @param segment Segment providing the phi planes.
  /// @param largePhiCutout Whether the excluded wedge is larger than pi.
  /// @param point Query point.
  /// @return `true` when the point is inside the infinite wedge.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool InPhiCutoutWedge(ZSegment const &segment,
                                                                            bool largePhiCutout,
                                                                            Vector3D<Precision> const &point);

  /// @brief Inside kernel for a known z/phi segment pair.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Inside_t InsideSegPhi(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex,
                               int phiIndex);

  /// @brief Inside kernel for repeated-z border segments.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Inside_t InsideSegBorder(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex);

  /// @brief Surface normal kernel.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool NormalKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                           Vector3D<Precision> &normal);

  /// @brief Public contains entry point for unplaced helpers.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContains(UnplacedStruct_t const &unplaced,
                                                                            Vector3D<Real_v> const &point,
                                                                            Bool_v &inside);

  /// @brief Public contains entry point.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside);

  /// @brief Public inside entry point.
  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside);

  /// @brief Public `DistanceToIn` entry point.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance);

  /// @brief Public `DistanceToOut` entry point.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance);

  /// @brief Public `SafetyToIn` entry point.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety);

  /// @brief Public `SafetyToOut` entry point.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety);

}; // End struct PolyhedronImplementation

namespace {

/// @brief Select the bounding-tube helper used by polyhedron bounds checks.
/// @details The current implementation keeps `UniversalTube` for both cases so
///          phi-limited hollow polyhedra preserve the historical behavior.

// SW (19.6.2015): switching to UniversalTube as Phi section was not
// correctly treated with a hollow tube
// TODO: this could be CORRECTLY put back for optimization
template <Polyhedron::EInnerRadii innerRadiiT>
struct HasInnerRadiiTraits {
  /// @brief Tube helper type used when inner radii are present.
  typedef TubeImplementation<TubeTypes::UniversalTube> TubeKernels;
};

template <>
struct HasInnerRadiiTraits<Polyhedron::EInnerRadii::kFalse> {
  /// @brief Tube helper type used when inner radii are absent.
  typedef TubeImplementation<TubeTypes::UniversalTube> TubeKernels;
};

/// @brief Decide whether inner radii participate in the current query.
/// @param hasInnerRadius Runtime segment flag.
/// @return `true` when inner radii must be tested.
template <Polyhedron::EInnerRadii innerRadiiT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatInner(bool hasInnerRadius)
{
  return hasInnerRadius;
}

/// @brief Specialized `TreatInner` for solids without inner radii.
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatInner<Polyhedron::EInnerRadii::kFalse>(bool /*hasInnerRadius*/)
{
  return false;
}

/// @brief Decide whether phi-cutout handling participates in the current query.
/// @param hasPhiCutout Runtime solid flag.
/// @return `true` when phi-cutout checks must be evaluated.
template <Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatPhi(bool /*hasPhiCutout*/)
{
  return true;
}

/// @brief Specialized `TreatPhi` for full-phi polyhedra.
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatPhi<Polyhedron::EPhiCutout::kFalse>(bool /*hasPhiCutout*/)
{
  return false;
}

/// @brief Specialized `TreatPhi` for runtime-selected phi cutouts.
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatPhi<Polyhedron::EPhiCutout::kGeneric>(bool hasPhiCutout)
{
  return hasPhiCutout;
}

/// @brief Decide whether the excluded phi wedge is larger than pi.
/// @param largePhiCutout Runtime solid flag.
/// @return `true` when the cutout exceeds pi.
template <Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool LargePhiCutout(bool largePhiCutout)
{
  return largePhiCutout;
}

/// @brief Specialized large-cutout selector for templates known to be small.
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool LargePhiCutout<Polyhedron::EPhiCutout::kTrue>(bool /*largePhiCutout*/)
{
  return false;
}

/// @brief Specialized large-cutout selector for templates known to exceed pi.
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool LargePhiCutout<Polyhedron::EPhiCutout::kLarge>(
    bool /*largePhiCutout*/)
{
  return true;
}

} // End anonymous namespace

namespace {

/// @brief Scalar helper locating a z-segment from the raw z-plane array.
/// @param pointZ Query z coordinate.
/// @param begin Pointer to the first z plane.
/// @param size Number of z planes.
/// @return Segment index convention used by `FindZSegment`.
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE int FindZSegmentKernel(Real_v const &pointZ, Precision const *begin,
                                                                    size_t size);
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE int FindZSegmentKernel<Precision>(Precision const &pointZ,
                                                                               Precision const *begin, size_t size)
{
  int index            = size - 1;
  Precision const *end = begin + index;
  // Modified algorithm to select the first section the position is close to
  // within boundary tolerance. This is important for degenerated Z polyhedra
  while (begin <= end && pointZ < *end + kTolerance) {
    --index;
    --end;
  }
  if ((size_t(index + 2) < size) && (pointZ > *(end + 1) - kTolerance)) return (index + 1);
  return index;
}
} // End anonymous namespace

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE int PolyhedronImplementation<innerRadiiT, phiCutoutT>::FindZSegment(
    UnplacedStruct_t const &unplaced, Real_v const &pointZ)
{
  return FindZSegmentKernel<Real_v>(pointZ, &unplaced.fZPlanes[0], unplaced.fZPlanes.size());
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE int PolyhedronImplementation<innerRadiiT, phiCutoutT>::FindPhiSegment(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{

  // Bounds between phi sections are represented as planes through the origin,
  // with the normal pointing along the phi direction.
  // To find the correct section, the point is projected onto each plane. If the
  // point is in front of a plane, but behind the subsequent plane, it must be
  // between them.

  int index                           = -1;
  SOA3D<Precision> const &phiSections = unplaced.fPhiSections;
  Real_v projectionFirst, projectionSecond;
  projectionFirst = point[0] * phiSections.x(0) + point[1] * phiSections.y(0) + point[2] * phiSections.z(0);
  for (int i = 1, iMax = unplaced.fSideCount + 1; i < iMax; ++i) {
    projectionSecond = point[0] * phiSections.x(i) + point[1] * phiSections.y(i) + point[2] * phiSections.z(i);
    if (projectionFirst > -kTolerance && projectionSecond < kTolerance) {
      index = i - 1;
      break;
    }
    projectionFirst = projectionSecond;
  }

  return index;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToInZSegment(
    UnplacedStruct_t const &unplaced, int segmentIndex, Vector3D<Real_v> const &point,
    Vector3D<Real_v> const &direction)
{

  Real_v distance;
  bool done = false;

  ZSegment const &segment = unplaced.fZSegments[segmentIndex];

  // If the outer shell is hit, this will always be the correct result
  distance = segment.outer.DistanceToIn<Real_v, false>(point, direction);
  done     = distance < InfinityLength<Real_v>();
  if (done) return distance;

  // If the outer shell is not hit and the phi cutout sides are hit, this will
  // always be the correct result
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    if (!done) distance = segment.phi.DistanceToIn<Real_v, false>(point, direction);
    if (unplaced.fHasLargePhiCutout) {
      // NOTE: The statement above is NOT always true: if fHasLargePhiCutout is false there can be a first hit of the
      // inner surface coming from the endcap holes
      done |= distance < InfinityLength<Real_v>();
      if (done) return distance;
    }
  }

  // Finally treat inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    Real_v distrmin = segment.inner.DistanceToIn<Real_v, true>(point, direction);
    if (!done && distance > distrmin) distance = distrmin;
  }

  return distance;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToOutZSegment(
    UnplacedStruct_t const &unplaced, int segmentIndex, Precision zMin, Precision zMax, Vector3D<Real_v> const &point,
    Vector3D<Real_v> const &direction)
{

  bool done       = false;
  Real_v distance = InfinityLength<Real_v>();

  ZSegment const &segment = unplaced.fZSegments[segmentIndex];

  // Check inner shell first, as it would always be the correct result
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    distance = segment.inner.DistanceToIn<Real_v, false>(point, direction);
    // Even if an inner surface is hit, there may be a phi hit before if there is no large phi cut
    if (unplaced.fHasLargePhiCutout) {
      done = distance < InfinityLength<Real_v>();
      if (done) return distance;
    }
  }

  // Check phi cutout if necessary. It is also possible to return here if a
  // result is found
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    Real_v distphi = segment.phi.DistanceToIn<Real_v, true>(point, direction);
    if (distance > distphi) distance = distphi;
  }

  done = distance > -kTolerance && distance < InfinityLength<Real_v>();
  if (done) return distance;

  // Finally check outer shell
  Real_v distout = segment.outer.DistanceToOut<Real_v>(point, direction, zMin, zMax);
  if (!done) distance = distout;

  return distance;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Precision
PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToZSegmentSquared(UnplacedStruct_t const &unplaced,
                                                                           int segmentIndex, int &phiIndex,
                                                                           Vector3D<Precision> const &point,
                                                                           bool pt_inside, int &iSurf)
{

  ZSegment const &segment = unplaced.fZSegments[segmentIndex];
  bool in_cutout          = phiIndex < 0;

  Precision safetySquared = InfinityLength<Precision>();
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout) && segment.phi.size() == 2) {
    //  Check if points is in the cutout wedge first.
    if (pt_inside || in_cutout) {
      // If point is in the cutout or if the call comes from SafetyToOut we need to check both phi planes
      iSurf         = 0;
      safetySquared = segment.phi.ScalarDistanceSquared(0, point);
      Precision saf = segment.phi.ScalarDistanceSquared(1, point);
      if (saf < safetySquared) {
        safetySquared = saf;
        iSurf         = 1;
      }
      // If the point is within the phi cutout wedge, we still need to check the
      // inner part
      if (in_cutout) {
        if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
          if (segment.inner.size() > 0) {
            Precision safetySquaredInner = segment.inner.ScalarDistanceSquared(0, point);
            if (safetySquaredInner < safetySquared) {
              iSurf         = 2;
              phiIndex      = 0;
              safetySquared = safetySquaredInner;
            }
            if (segment.inner.size() > 1) {
              safetySquaredInner = segment.inner.ScalarDistanceSquared(segment.inner.size() - 1, point);
              if (safetySquaredInner < safetySquared) {
                iSurf         = 2;
                phiIndex      = segment.inner.size() - 1;
                safetySquared = safetySquaredInner;
              }
            }
          }
        }
        return safetySquared;
      }
    }
  }

  if (in_cutout && segmentIndex > 0 && segmentIndex < unplaced.fZSegments.size() - 1 &&
      unplaced.fZPlanes[segmentIndex] == unplaced.fZPlanes[segmentIndex + 1]) {
    // We are checking a segment at same Z. We have to check the inner and outer
    // quadrilaterals for first and last phi
    Precision safetySquaredOuter = InfinityLength<Precision>();
    if (segment.outer.size() > 0) {
      safetySquaredOuter = segment.outer.ScalarDistanceSquared(0, point);
      if (safetySquaredOuter < safetySquared) {
        iSurf         = 3;
        phiIndex      = 0;
        safetySquared = safetySquaredOuter;
      }
      if (segment.outer.size() > 1) {
        safetySquaredOuter = segment.outer.ScalarDistanceSquared(segment.outer.size() - 1, point);
        if (safetySquaredOuter < safetySquared) {
          iSurf         = 3;
          phiIndex      = segment.outer.size() - 1;
          safetySquared = safetySquaredOuter;
        }
      }
    }
    Precision safetySquaredInner = InfinityLength<Precision>();
    if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
      if (segment.inner.size() > 0) {
        safetySquaredInner = segment.inner.ScalarDistanceSquared(0, point);
        if (safetySquaredInner < safetySquared) {
          iSurf         = 2;
          phiIndex      = 0;
          safetySquared = safetySquaredInner;
        }
        if (segment.inner.size() > 1) {
          safetySquaredInner = segment.inner.ScalarDistanceSquared(segment.inner.size() - 1, point);
          if (safetySquaredInner < safetySquared) {
            iSurf         = 2;
            phiIndex      = segment.inner.size() - 1;
            safetySquared = safetySquaredInner;
          }
        }
      }
    }
    return safetySquared;
  }

  // Otherwise check the outer shell
  // TODO: we need to check segment.outer.size() > 0
  Precision safetySquaredOuter = InfinityLength<Precision>();
  if (segment.outer.size() > 0) safetySquaredOuter = segment.outer.ScalarDistanceSquared(phiIndex, point);

  // And finally the inner
  Precision safetySquaredInner = InfinityLength<Precision>();
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    if (segment.inner.size() > 0) safetySquaredInner = segment.inner.ScalarDistanceSquared(phiIndex, point);
  }
  if (safetySquaredInner < safetySquared) {
    iSurf         = 2;
    safetySquared = safetySquaredInner;
  }
  if (safetySquaredOuter < safetySquared) {
    iSurf         = 3;
    safetySquared = safetySquaredOuter;
  }
  return safetySquared;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <bool pointInsideT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToEndcaps(
    UnplacedStruct_t const &unplaced, bool /*goingRight*/, Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction, Precision &distance)
{

  ZSegment const *segment;
  Precision zPlane;

  // Determine whether to use first segment/first endcap or last segment/second
  // endcap
  // NOTE: might make this more elegant
  if (pointInsideT) // inside version
  {
    if (direction[2] < 0) {
      segment = &unplaced.fZSegments[0];
      zPlane  = unplaced.fZPlanes[0];
    } else {
      segment = &unplaced.fZSegments[unplaced.fZSegments.size() - 1];
      zPlane  = unplaced.fZPlanes[unplaced.fZSegments.size()];
    }
  } else // outside version
  {
    if (direction[2] < 0) {
      segment = &unplaced.fZSegments[unplaced.fZSegments.size() - 1];
      zPlane  = unplaced.fZPlanes[unplaced.fZSegments.size()];
    } else {
      segment = &unplaced.fZSegments[0];
      zPlane  = unplaced.fZPlanes[0];
    }
  }

  Precision distanceTest = (zPlane - point[2]) / NonZero(direction[2]);
  // If the distance is not better there's no reason to check for validity
  if (distanceTest < -kTolerance || distanceTest >= distance) return;

  Vector3D<Precision> intersection = point + distanceTest * direction;
  // Intersection point must be inside outer shell and outside inner shell
  if (!segment->outer.Contains<Precision>(intersection)) return;
  if (TreatInner<innerRadiiT>(segment->hasInnerRadius())) {
    if (segment->inner.Contains<Precision>(intersection)) return;
  }
  // Intersection point must not be in phi cutout wedge
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    if (InPhiCutoutWedge(*segment, unplaced.fHasLargePhiCutout, intersection)) {
      return;
    }
  }

  distance = distanceTest;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<
    innerRadiiT, phiCutoutT>::SafetyToEndcapsSquared(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                                     Precision &distanceSquared, int &iz)
{

  // Compute both distances (simple subtractions) to determine which is closer
  Precision firstDistance = unplaced.fZPlanes[0] - point[2];
  Precision lastDistance  = unplaced.fZPlanes[unplaced.fZSegments.size()] - point[2];

  // Only treat the closest endcap
  bool isFirst            = Abs(firstDistance) < Abs(lastDistance);
  iz                      = 0;
  ZSegment const &segment = isFirst ? unplaced.fZSegments[0] : unplaced.fZSegments[unplaced.fZSegments.size() - 1];

  Precision distanceTest        = isFirst ? firstDistance : lastDistance;
  Precision distanceTestSquared = distanceTest * distanceTest;
  // No need to investigate further if distance is larger anyway
  if (distanceTestSquared >= distanceSquared) return;

  // Check if projection is within the endcap bounds
  Vector3D<Precision> intersection(point[0], point[1], point[2] + distanceTest);
  if (!segment.outer.Contains<Precision>(intersection)) return;
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    if (segment.inner.Contains<Precision>(intersection)) return;
  }
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    if (InPhiCutoutWedge(segment, unplaced.fHasLargePhiCutout, intersection)) {
      return;
    }
  }

  iz              = (isFirst) ? -1 : 1;
  distanceSquared = distanceTestSquared;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool PolyhedronImplementation<innerRadiiT, phiCutoutT>::InPhiCutoutWedge(
    ZSegment const &segment, bool largePhiCutout, Vector3D<Precision> const &point)
{
  bool pointSeg0 = point.Dot(segment.phi.GetNormal(0)) + segment.phi.GetDistance(0) >= kTolerance;
  bool pointSeg1 = point.Dot(segment.phi.GetNormal(1)) + segment.phi.GetDistance(1) >= kTolerance;
  // For a cutout larger than 180 degrees, the point is in the wedge if it is
  // in front of at least one plane.
  if (LargePhiCutout<phiCutoutT>(largePhiCutout)) {
    return pointSeg0 || pointSeg1;
  }
  // Otherwise it should be in front of both planes
  return pointSeg0 && pointSeg1;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE Inside_t PolyhedronImplementation<innerRadiiT, phiCutoutT>::InsideSegBorder(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex)
{
  // Check Inside if the point is in between two non-continuous "border-like"
  // segments. The zIndex corresponds to the lesser index of the 2 planes having the same Z.
  // The Quadrilaterals algorithm for Inside in this case does not work.

  ZSegment const &segment = unplaced.fZSegments[zIndex];
  // Identify phi index
  int phiIndex = FindPhiSegment<Precision>(unplaced, point);
  if (phiIndex < 0) return EInside::kOutside;
  // Get the vector perpendicular to the rmax edge of the outer quadrilateral
  Vector3D<Precision> const &vout = (segment.outer.size()) ? segment.outer.GetSideVectors()[0].GetNormals()[phiIndex]
                                                           : segment.inner.GetSideVectors()[0].GetNormals()[phiIndex];
  // Compute the projection of the point vectoron the vout vector. This
  // corresponds to a "radius" or the point.
  Precision rdotvout = vecCore::math::Abs<Precision>(point.Dot(vout));
  // Now compare the point radius with the ranges corresponding to the lower
  // and upper segments
  bool in1 = (rdotvout > unplaced.fRMin[zIndex] - kTolerance) && (rdotvout < unplaced.fRMax[zIndex] + kTolerance);
  bool in2 =
      (rdotvout > unplaced.fRMin[zIndex + 1] - kTolerance) && (rdotvout < unplaced.fRMax[zIndex + 1] + kTolerance);
  if (in1 && in2) {
    if ((rdotvout < unplaced.fRMin[zIndex] + kTolerance) || (rdotvout > unplaced.fRMax[zIndex] - kTolerance) ||
        (rdotvout < unplaced.fRMin[zIndex + 1] + kTolerance) || (rdotvout > unplaced.fRMax[zIndex + 1] - kTolerance))
      return EInside::kSurface;
    // Need to check phi surface
    if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
      Inside_t insidePhi = unplaced.fPhiWedge.Inside<Precision, Inside_t>(point);
      return insidePhi;
    }
    return EInside::kInside;
  }
  if (!in1 && !in2) return EInside::kOutside;
  return EInside::kSurface;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE Inside_t PolyhedronImplementation<innerRadiiT, phiCutoutT>::InsideSegPhi(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex, int phiIndex)
{
  // Check inside for a specified z segment and phi edge
  if (phiIndex < 0) return EInside::kOutside;

  // Z range
  Precision dz = vecCore::math::Abs(point[2] - unplaced.fBoundingTubeOffset) -
                 0.5 * (unplaced.fZPlanes[unplaced.fZSegments.size()] - unplaced.fZPlanes[0]);
  //  if (vecCore::math::Abs(dz) < kHalfTolerance) return EInside::kSurface;
  if (dz > kHalfTolerance) return EInside::kOutside;

  if (unplaced.fSameZ[zIndex]) return InsideSegBorder(unplaced, point, zIndex);

  ZSegment const &segment = unplaced.fZSegments[zIndex];

  // Check that the point is in the outer shell
  {
    Inside_t insideOuter = segment.outer.Inside<Precision, Inside_t>(point, phiIndex);
    if (insideOuter != EInside::kInside) return insideOuter;
  }

  // Check that the point is not in the inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    Inside_t insideInner = segment.inner.Inside<Precision, Inside_t>(point, phiIndex);
    if (insideInner == EInside::kInside) return EInside::kOutside;
    if (insideInner == EInside::kSurface) return EInside::kSurface;
  }

  // Check that the point is not in the phi cutout wedge
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    Inside_t insidePhi = unplaced.fPhiWedge.Inside<Precision, Inside_t>(point);
    if (insidePhi != EInside::kInside) return insidePhi;
  }

  if (vecCore::math::Abs(dz) < kHalfTolerance) return EInside::kSurface;
  return EInside::kInside;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE bool PolyhedronImplementation<innerRadiiT, phiCutoutT>::NormalKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, Vector3D<Precision> &normal)
{

  Precision safety = InfinityLength<Precision>();
  const int zMax   = unplaced.fZSegments.size();
  int zIndex       = FindZSegment<Precision>(unplaced, point[2]);
  if (zIndex < 0) {
    normal = Vector3D<Precision>(0, 0, -1);
    return true;
  }

  if (zIndex >= zMax) {
    normal = Vector3D<Precision>(0, 0, 1);
    return true;
  }

  int iSeg = zIndex;
  Precision dz;
  int iSurf    = -1;
  int iz       = 0;
  int phiIndex = FindPhiSegment<Precision>(unplaced, point);

  // Right
  for (int z = zIndex; z < zMax;) {
    int iSurfCrt        = -1;
    Precision safetySeg = SafetyToZSegmentSquared(unplaced, z, phiIndex, point, true, iSurfCrt);
    if (safetySeg < safety) {
      safety = safetySeg;
      iSeg   = z;
      iSurf  = iSurfCrt;
    }
    ++z;
    dz = unplaced.fZPlanes[z] - point[2];
    if (dz * dz > safety) break;
  }
  // Left
  for (int z = zIndex - 1; z >= 0; --z) {
    int iSurfCrt        = -1;
    Precision safetySeg = SafetyToZSegmentSquared(unplaced, z, phiIndex, point, true, iSurfCrt);
    if (safetySeg < safety) {
      safety = safetySeg;
      iSeg   = z;
      iSurf  = iSurfCrt;
    }
    dz = point[2] - unplaced.fZPlanes[z];
    if (dz * dz > safety) break;
  }

  // Endcap
  SafetyToEndcapsSquared(unplaced, point, safety, iz);
  if (iz != 0) {
    normal = Vector3D<Precision>(0, 0, iz);
    return true;
  }

  // Retrieve the segment the point is closest to.
  ZSegment const &segment = unplaced.fZSegments[iSeg];
  if (iSurf >= 0 && iSurf < 2) {
    normal = segment.phi.GetNormal(iSurf);
  } else {
    if (iSurf == 2)
      normal = -1. * segment.inner.GetNormal(phiIndex);
    else
      normal = segment.outer.GetNormal(phiIndex);
  }
  return true;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v, typename Bool_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::UnplacedContains(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
{
  Contains(unplaced, point, inside);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v, typename Bool_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::Contains(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
{
  Vector3D<Precision> localPoint(point[0], point[1], point[2]);

  // First check if in bounding tube.
  {
    bool inBounds;
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template Contains<>(
        unplaced.fBoundingTube,
        Vector3D<Precision>(localPoint[0], localPoint[1], localPoint[2] - unplaced.fBoundingTubeOffset), inBounds);
    if (!inBounds) {
      inside = false;
      return;
    }
  }

  // Find correct segment by checking Z-bounds.
  int zIndex = FindZSegment<Precision>(unplaced, localPoint[2]);
  if (!(zIndex >= 0 && zIndex < unplaced.fZSegments.size())) {
    inside = false;
    return;
  }

  ZSegment const &segment = unplaced.fZSegments[zIndex];

  // In case the point lies at the same Z as 2 consecutive planes, the lesser
  // index is selected. The Quadrilaterals algorithm for Contains in this case
  // does not work.
  if (unplaced.fSameZ[zIndex]) {
    int phiIndex = FindPhiSegment<Precision>(unplaced, localPoint);
    if (phiIndex < 0) {
      inside = false;
      return;
    }
    Vector3D<Precision> const &vout = (segment.outer.size()) ? segment.outer.GetSideVectors()[0].GetNormals()[phiIndex]
                                                             : segment.inner.GetSideVectors()[0].GetNormals()[phiIndex];
    Precision rdotvout              = vecCore::math::Abs<Precision>(localPoint.Dot(vout));
    bool in1                        = (rdotvout >= unplaced.fRMin[zIndex]) && (rdotvout <= unplaced.fRMax[zIndex]);
    bool in2 = (rdotvout >= unplaced.fRMin[zIndex + 1]) && (rdotvout <= unplaced.fRMax[zIndex + 1]);
    inside   = (in1 | in2);
    return;
  }

  if (!segment.outer.Contains<Precision>(localPoint)) {
    inside = false;
    return;
  }

  if (TreatInner<innerRadiiT>(segment.hasInnerRadius()) && segment.inner.Contains<Precision>(localPoint)) {
    inside = false;
    return;
  }

  // The bounding tube already handles phi in principle, but the dedicated
  // polyhedron cutout check still matters because it uses different tolerances.
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout) && !segment.phi.Contains<Precision>(localPoint)) {
    inside = false;
    return;
  }

  inside = true;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v, typename Inside_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::Inside(UnplacedStruct_t const &unplaced,
                                                                                       Vector3D<Real_v> const &point,
                                                                                       Inside_v &inside)
{
  Vector3D<Precision> localPoint(point[0], point[1], point[2]);

  {
    bool inBounds;
    // The bounding tube is intentionally slightly enlarged to keep the fast
    // reject compatible with the tolerance-sensitive inside checks below. The
    // old ideal tube was too tight for early rejection because this path must
    // preserve the same tolerance envelope as the detailed segment checks.
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template Contains<>(
        unplaced.fBoundingTube,
        Vector3D<Precision>(localPoint[0], localPoint[1], localPoint[2] - unplaced.fBoundingTubeOffset), inBounds);
    if (!inBounds) {
      inside = EInside::kOutside;
      return;
    }
  }

  // For two consecutive planes with identical Z, FindZSegment must prefer the
  // first segment so the degenerate border segment is checked before its
  // neighbor. The enlarged bounding tube can still accept points outside the
  // actual Z range, so the explicit segment-range rejection remains necessary.
  int zIndex = FindZSegment<Precision>(unplaced, localPoint[2]);
  if (zIndex < 0 || zIndex > (unplaced.fZSegments.size() - 1)) {
    inside = EInside::kOutside;
    return;
  }

  ZSegment const &segment = unplaced.fZSegments[zIndex];
  if (unplaced.fSameZ[zIndex]) {
    inside = InsideSegBorder(unplaced, localPoint, zIndex);
    return;
  }

  {
    Inside_t insideOuter = segment.outer.Inside<Precision, Inside_t>(localPoint);
    if (insideOuter != EInside::kInside) {
      inside = insideOuter;
      return;
    }
  }

  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    Inside_t insideInner = segment.inner.Inside<Precision, Inside_t>(localPoint);
    if (insideInner == EInside::kInside) {
      inside = EInside::kOutside;
      return;
    }
    if (insideInner == EInside::kSurface) {
      inside = EInside::kSurface;
      return;
    }
  }

  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    Inside_t insidePhi = segment.phi.Inside<Precision, Inside_t>(localPoint);
    if (insidePhi != EInside::kInside) {
      inside = insidePhi;
      return;
    }
  }

  // After the radial and phi checks, the point can still be on the global Z
  // endcap boundary. Classify that tolerance band as surface rather than inside.
  Precision dz = vecCore::math::Abs(vecCore::math::Abs(localPoint[2] - unplaced.fBoundingTubeOffset) -
                                    0.5 * (unplaced.fZPlanes[unplaced.fZSegments.size()] - unplaced.fZPlanes[0]));
  inside       = dz < kTolerance ? EInside::kSurface : EInside::kInside;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToIn(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{
  Vector3D<Precision> localPoint(point[0], point[1], point[2]);
  Vector3D<Precision> localDirection(direction[0], direction[1], direction[2]);
  Precision localStepMax = stepMax;

  if ((localPoint[2] < unplaced.fZPlanes[0] + kTolerance) && localDirection[2] <= 0) {
    distance = InfinityLength<Real_v>();
    return;
  }
  if ((localPoint[2] > unplaced.fZPlanes[unplaced.fZSegments.size()] - kTolerance) && localDirection[2] >= 0) {
    distance = InfinityLength<Real_v>();
    return;
  }

  // Explicitly detect wrong-side points. Without this guard, a point already
  // inside the solid could be reported as having a normal entry distance.
  Inside_t insideState;
  Inside(unplaced, localPoint, insideState);
  if (insideState == kInside) {
    distance = Real_v(-1.);
    return;
  }

  bool inBounds;
  Precision tubeDistance = 0.;
  {
    Vector3D<Precision> boundsPoint(localPoint[0], localPoint[1], localPoint[2] - unplaced.fBoundingTubeOffset);
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template Contains<>(unplaced.fBoundingTube, boundsPoint, inBounds);
    // If the point is inside the bounding tube, its DistanceToIn value is not
    // reliable for rejecting the ray; the tube can report a forward boundary
    // even though the polyhedron entry must be decided by the detailed segments.
    if (!inBounds) {
      // When the point is outside the bounding tube, a missed tube entry is a
      // valid early rejection for the whole polyhedron.
      HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template DistanceToIn<>(
          unplaced.fBoundingTube, boundsPoint, localDirection, localStepMax, tubeDistance);
      if (tubeDistance == InfinityLength<Precision>()) {
        distance = InfinityLength<Real_v>();
        return;
      }
    }
  }

  int zIndex     = FindZSegment<Precision>(unplaced, localPoint[2]);
  const int zMax = unplaced.fZSegments.size();
  // Clamp only after FindZSegment: the first or last Z segment still has to be
  // checked even when the start point is outside the nominal Z range.
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  bool goingRight = localDirection[2] >= 0;

  Precision result = InfinityLength<Precision>();
  if (goingRight) {
    for (int zSegCount = unplaced.fZSegments.size(); zIndex < zSegCount; ++zIndex) {
      result = DistanceToInZSegment<Precision>(unplaced, zIndex, localPoint, localDirection);
      // Once a valid segment hit is found, farther Z segments cannot provide a
      // shorter entry; only the endcaps still need to be minimized.
      if (result >= 0 && result < InfinityLength<Precision>()) break;
    }
  } else {
    for (; zIndex >= 0; --zIndex) {
      result = DistanceToInZSegment<Precision>(unplaced, zIndex, localPoint, localDirection);
      // Once a valid segment hit is found, farther Z segments cannot provide a
      // shorter entry; only the endcaps still need to be minimized.
      if (result >= 0 && result < InfinityLength<Precision>()) break;
    }
  }

  // Endcaps are not part of the Z-segment side traversal, so minimize them
  // explicitly. The final tube comparison is only a sanity check against the
  // coarse bounding volume, not the geometry algorithm that finds the entry.
  DistanceToEndcaps<false>(unplaced, goingRight, localPoint, localDirection, result);
  result   = (result >= tubeDistance - 1E-6) ? result : vecgeom::InfinityLength<Precision>();
  distance = result;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{
  // Keep stepMax ignored until the finite-limit DistanceToOut convention is
  // revised; clamping an error-like infinity to stepMax would hide missed exits.
  (void)stepMax;
  Vector3D<Precision> localPoint(point[0], point[1], point[2]);
  Vector3D<Precision> localDirection(direction[0], direction[1], direction[2]);

  const int zMax = unplaced.fZSegments.size();
  if ((localPoint[2] < unplaced.fZPlanes[0] - kTolerance) || (localPoint[2] > unplaced.fZPlanes[zMax] + kTolerance)) {
    distance = Real_v(-1.);
    return;
  }

  Inside_t insideState;
  Inside(unplaced, localPoint, insideState);
  if (insideState == kOutside) {
    distance = Real_v(-1.);
    return;
  }

  int zIndex = FindZSegment<Precision>(unplaced, localPoint[2]);
  zIndex     = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  bool goingRight = localDirection[2] >= 0;

  Precision result = InfinityLength<Precision>();
  if (goingRight) {
    for (; zIndex < zMax; ++zIndex) {
      result = DistanceToOutZSegment<Precision>(unplaced, zIndex, unplaced.fZPlanes[zIndex],
                                                unplaced.fZPlanes[zIndex + 1], localPoint, localDirection);
      if (result >= 0 && result < InfinityLength<Precision>()) break;
      if (unplaced.fZPlanes[zIndex] - localPoint[2] > result) break;
    }
  } else {
    for (; zIndex >= 0; --zIndex) {
      result = DistanceToOutZSegment<Precision>(unplaced, zIndex, unplaced.fZPlanes[zIndex],
                                                unplaced.fZPlanes[zIndex + 1], localPoint, localDirection);
      if (result >= 0 && result < InfinityLength<Precision>()) break;
      if (localPoint[2] - unplaced.fZPlanes[zIndex] > result) break;
    }
  }

  if (Abs(localDirection[2]) > kTolerance) {
    DistanceToEndcaps<true>(unplaced, goingRight, localPoint, localDirection, result);
  }

  if (result >= InfinityLength<Precision>()) result = 0.;
  distance = result;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToIn(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{
  Vector3D<Precision> localPoint(point[0], point[1], point[2]);

  Precision result = InfinityLength<Precision>();
  Precision dz;
  int iSurf, iz;

  const int zMax = unplaced.fZSegments.size();
  int zIndex     = FindZSegment<Precision>(unplaced, localPoint[2]);
  zIndex         = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  int phiIndex = FindPhiSegment<Precision>(unplaced, localPoint);

  Inside_t insideState = InsideSegPhi(unplaced, localPoint, zIndex, phiIndex);
  if (insideState == EInside::kSurface) {
    safety = Real_v(0.);
    return;
  }
  bool contains = (insideState == EInside::kInside);
  if (contains) {
    safety = Real_v(-1.);
    return;
  }

  for (int z = zIndex; z < zMax;) {
    result = Min(result, SafetyToZSegmentSquared(unplaced, z, phiIndex, localPoint, false, iSurf));
    ++z;
    dz = unplaced.fZPlanes[z] - localPoint[2];
    if (dz * dz > result) break;
  }
  for (int z = zIndex - 1; z >= 0; --z) {
    result = Min(result, SafetyToZSegmentSquared(unplaced, z, phiIndex, localPoint, false, iSurf));
    dz     = localPoint[2] - unplaced.fZPlanes[z];
    if (dz * dz > result) break;
  }

  SafetyToEndcapsSquared(unplaced, localPoint, result, iz);
  safety = vecCore::math::Sqrt(result);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{
  Vector3D<Precision> localPoint(point[0], point[1], point[2]);

  Precision result = InfinityLength<Precision>();
  Precision dz;
  int iSurf, iz;

  const int zMax = unplaced.fZSegments.size();
  int zIndex     = FindZSegment<Precision>(unplaced, localPoint[2]);
  zIndex         = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  int phiIndex = FindPhiSegment<Precision>(unplaced, localPoint);

  Inside_t insideState = InsideSegPhi(unplaced, localPoint, zIndex, phiIndex);
  if (insideState == EInside::kSurface) {
    safety = Real_v(0.);
    return;
  }
  bool contains = (insideState == EInside::kInside);
  if (!contains) {
    safety = Real_v(-1.);
    return;
  }

  for (int z = zIndex; z < zMax;) {
    result = Min(result, SafetyToZSegmentSquared(unplaced, z, phiIndex, localPoint, true, iSurf));
    ++z;
    dz = unplaced.fZPlanes[z] - localPoint[2];
    if (dz * dz > result) break;
  }
  for (int z = zIndex - 1; z >= 0; --z) {
    result = Min(result, SafetyToZSegmentSquared(unplaced, z, phiIndex, localPoint, true, iSurf));
    dz     = localPoint[2] - unplaced.fZPlanes[z];
    if (dz * dz > result) break;
  }

  SafetyToEndcapsSquared(unplaced, localPoint, result, iz);
  safety = vecCore::math::Sqrt(result);
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
