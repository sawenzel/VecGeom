/// \file PolyhedronImplementation.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include "base/Global.h"

#include "backend/Backend.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/Backend.h"
#endif
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/kernel/TubeImplementation.h"
#include "volumes/Quadrilaterals.h"
#include "volumes/UnplacedPolyhedron.h"

namespace VECGEOM_NAMESPACE {

template <bool treatInnerT>
struct PolyhedronImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::int_v FindZSegment(
      UnplacedPolyhedron const &polyhedron,
      typename Backend::precision_v const &pointZ);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static int ScalarFindPhiSegment(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToInZSegment(
      UnplacedPolyhedron const &polyhedron,
      int segmentIndex,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToOutZSegment(
      UnplacedPolyhedron const &polyhedron,
      int segmentIndex,
      Precision zMin,
      Precision zMax,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision ScalarSafetyToZSegmentSquared(
      UnplacedPolyhedron const &polyhedron,
      int segmentIndex,
      int phiIndex,
      Vector3D<Precision> const &point);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ScalarDistanceToInEndcaps(
      UnplacedPolyhedron const &polyhedron,
      bool goingRight,
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction,
      Precision &distance);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ScalarDistanceToOutEndcaps(
      UnplacedPolyhedron const &polyhedron,
      bool goingRight,
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction,
      Precision &distance);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ScalarSafetyToEndcapsSquared(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &point,
      Precision &distance);

  /// \return Checks whether a point is within the infinite phi wedge formed
  ///         from origin in the cutout angle between the first and last vector.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v InPhiCutoutWedge(
      UnplacedPolyhedron::ZSegment const &segment,
      bool largePhiCutout,
      Vector3D<typename Backend::precision_v> const &point);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static bool ScalarContainsKernel(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &localPoint);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Inside_t ScalarInsideKernel(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &localPoint);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision ScalarDistanceToInKernel(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction,
      const Precision stepMax);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision ScalarDistanceToOutKernel(
      UnplacedPolyhedron const &unplaced,
      Vector3D<Precision> const &localPoint,
      Vector3D<Precision> const &localDirection,
      const Precision stepMax);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision ScalarSafetyKernel(
      UnplacedPolyhedron const &unplaced,
      Vector3D<Precision> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedPolyhedron const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedPolyhedron const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedPolyhedron const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

}; // End struct PolyhedronImplementation

namespace {

template <bool hasInnerRadiiT>
struct HasInnerRadiiTraits;

template <>
struct HasInnerRadiiTraits<true> {
  typedef TubeImplementation<translation::kIdentity,
      rotation::kIdentity, TubeTypes::HollowTube> TubeKernels;
};

template <>
struct HasInnerRadiiTraits<false> {
  typedef TubeImplementation<translation::kIdentity,
      rotation::kIdentity, TubeTypes::NonHollowTube> TubeKernels;
};

} // End anonymous namespace

namespace {

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::int_v FindZSegmentKernel(
    Precision const *begin,
    Precision const *end,
    typename Backend::precision_v const &pointZ);

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int FindZSegmentKernel<kScalar>(
    Precision const *begin,
    Precision const *end,
    Precision const &pointZ) {
  int index = -1;
  while (begin < end && pointZ > *begin) {
    ++index;
    ++begin;
  }
  return index;
}

#ifdef VECGEOM_CUDA
template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int FindZSegmentKernel<kCuda>(
    Precision const *begin,
    Precision const *end,
    Precision const &pointZ) {
  return FindZSegmentKernel<kScalar>(begin, end, pointZ);
}
#endif

} // End anonymous namespace

template <bool treatInnerT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v PolyhedronImplementation<treatInnerT>::FindZSegment(
    UnplacedPolyhedron const &polyhedron,
    typename Backend::precision_v const &pointZ) {
  return FindZSegmentKernel<Backend>(
      &polyhedron.GetZPlanes()[0],
      &polyhedron.GetZPlanes()[0]+polyhedron.GetZPlanes().size(),
      pointZ);
}

template <bool treatInnerT>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
int PolyhedronImplementation<treatInnerT>::ScalarFindPhiSegment(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &point) {

  SOA3D<Precision> const &phiSections = polyhedron.GetPhiSections();
  int i = 0;
  const int iMax = polyhedron.GetSideCount()-1;
  while (i < iMax) {
    bool inSection = point[0]*phiSections.x(i) + point[1]*phiSections.y(i) +
                     point[2]*phiSections.z(i) >= 0 &&
                     point[0]*phiSections.x(i+1) + point[1]*phiSections.y(i+1) +
                     point[2]*phiSections.z(i+1) < 0;
    if (inSection) break;
    ++i;
  }
  return i;
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<treatInnerT>::DistanceToInZSegment(
    UnplacedPolyhedron const &polyhedron,
    int segmentIndex,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t distance;
  Bool_t done;

  UnplacedPolyhedron::ZSegment const &segment =
      polyhedron.GetZSegment(segmentIndex);

  // If the outer shell is hit, it will always be the correct result
  distance = segment.outer.DistanceToIn<Backend, false>(point, direction);
  done = distance < kInfinity;
  if (IsFull(done)) return distance;

  if (polyhedron.HasPhiCutout()) {
    MaskedAssign(!done,
                 segment.phi.DistanceToIn<Backend, false>(point, direction),
                 &distance);
  }
  done = distance < kInfinity;
  if (IsFull(done)) return distance;

  // Inner shell
  if (treatInnerT && segment.hasInnerRadius) {
    MaskedAssign(!done,
                 segment.inner.DistanceToIn<Backend, true>(point, direction),
                 &distance);
  }

  return distance;
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<treatInnerT>::DistanceToOutZSegment(
    UnplacedPolyhedron const &polyhedron,
    int segmentIndex,
    Precision zMin,
    Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Bool_t done(false);
  Float_t distance(kInfinity);

  UnplacedPolyhedron::ZSegment const &segment =
      polyhedron.GetZSegment(segmentIndex);

  // Check inner shell first, as it would always be the correct result
  if (treatInnerT && segment.hasInnerRadius) {
    distance = segment.inner.DistanceToIn<Backend, false>(point, direction);
    done = distance < kInfinity;
    if (IsFull(done)) return distance;
  }

  // Check phi cutout if necessary. It is also possible to return here if a
  // result is found
  if (polyhedron.HasPhiCutout()) {
    MaskedAssign(!done,
                 segment.phi.DistanceToIn<Backend, true>(point, direction),
                 &distance);
    done = distance < kInfinity;
    if (IsFull(done)) return distance;
  }

  // Finally check outer shell
  MaskedAssign(!done,
               segment.outer.DistanceToOut<Backend>(
                  point, direction, zMin, zMax),
               &distance);

  return distance;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PolyhedronImplementation<treatInnerT>::ScalarSafetyToZSegmentSquared(
    UnplacedPolyhedron const &polyhedron,
    int segmentIndex,
    int phiIndex,
    Vector3D<Precision> const &point) {

  UnplacedPolyhedron::ZSegment const &segment =
      polyhedron.GetZSegment(segmentIndex);

  if (polyhedron.HasPhiCutout() &&
      InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                point)) {
    return Min(segment.phi.ScalarDistanceSquared(0, point),
               segment.phi.ScalarDistanceSquared(1, point));
  }

  Precision safetySquared =
      segment.outer.ScalarDistanceSquared(phiIndex, point);

  if (treatInnerT && segment.hasInnerRadius) {
    safetySquared = Min(safetySquared,
        segment.inner.ScalarDistanceSquared(phiIndex, point));
  }

  return safetySquared;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT>::ScalarDistanceToInEndcaps(
    UnplacedPolyhedron const &polyhedron,
    bool goingRight,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    Precision &distance) {

  UnplacedPolyhedron::ZSegment const *segment;
  Precision zPlane;
  if (goingRight && point[2] < polyhedron.GetZPlane(0)) {
    segment = &polyhedron.GetZSegment(0);
    zPlane = polyhedron.GetZPlane(0);
  } else if (!goingRight &&
             point[2] > polyhedron.GetZPlane(polyhedron.GetZSegmentCount())) {
    segment = &polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);
    zPlane = polyhedron.GetZPlane(polyhedron.GetZSegmentCount());
  } else {
    return;
  }

  Precision distanceTest = (zPlane - point[2]) / direction[2];
  if (distanceTest < 0 || distanceTest >= distance) return;

  Vector3D<Precision> intersection = point + distanceTest*direction;
  // Intersection point must be inside outer shell and outside inner shell
  if (!segment->outer.Contains<kScalar>(intersection)) return;
  if (treatInnerT && segment->hasInnerRadius) {
    if (segment->inner.Contains<kScalar>(intersection)) return;
  }
  if (polyhedron.HasPhiCutout()) {
    if (InPhiCutoutWedge<kScalar>(*segment, polyhedron.HasLargePhiCutout(),
                                  intersection)) {
      return;
    }
  }

  distance = distanceTest;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT>::ScalarDistanceToOutEndcaps(
    UnplacedPolyhedron const &polyhedron,
    bool goingRight,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    Precision &distance) {

  UnplacedPolyhedron::ZSegment const *segment;
  Precision zPlane;
  if (!goingRight && point[2] > polyhedron.GetZPlane(0)) {
    segment = &polyhedron.GetZSegment(0);
    zPlane = polyhedron.GetZPlane(0);
  } else if (goingRight &&
             point[2] < polyhedron.GetZPlane(polyhedron.GetZSegmentCount())) {
    segment = &polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);
    zPlane = polyhedron.GetZPlane(polyhedron.GetZSegmentCount());
  } else {
    return;
  }

  Precision distanceTest = (zPlane - point[2]) / direction[2];
  if (distanceTest < 0 || distanceTest >= distance) return;

  Vector3D<Precision> intersection = point + distanceTest*direction;
  // Intersection point must be inside outer shell and outside inner shell
  if (!segment->outer.Contains<kScalar>(intersection)) return;
  if (treatInnerT && segment->hasInnerRadius) {
    if (segment->inner.Contains<kScalar>(intersection)) return;
  }
  if (polyhedron.HasPhiCutout()) {
    if (InPhiCutoutWedge<kScalar>(*segment, polyhedron.HasLargePhiCutout(),
                                  intersection)) {
      return;
    }
  }

  distance = distanceTest;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT>::ScalarSafetyToEndcapsSquared(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &point,
    Precision &distanceSquared) {

  Precision firstDistance = polyhedron.GetZPlane(0) - point[2];
  Precision lastDistance =
      polyhedron.GetZPlane(polyhedron.GetZSegmentCount()) - point[2];

  bool isFirst = Abs(firstDistance) < Abs(lastDistance);
  UnplacedPolyhedron::ZSegment const &segment =
      isFirst ? polyhedron.GetZSegment(0)
              : polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);

  Precision distanceTest = isFirst ? firstDistance : lastDistance;
  Precision distanceTestSquared = distanceTest*distanceTest;
  // No need to investigate further if distance is larger anyway
  if (distanceTestSquared >= distanceSquared) return;

  // Check if projection is within the endcap bounds
  Vector3D<Precision> intersection(point[0], point[1], point[2] + distanceTest);
  if (!segment.outer.Contains<kScalar>(intersection)) return;
  if (treatInnerT && segment.hasInnerRadius) {
    if (segment.inner.Contains<kScalar>(intersection)) return;
  }
  if (polyhedron.HasPhiCutout()) {
    if (InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                  intersection)) {
      return;
    }
  }

  distanceSquared = distanceTestSquared;
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::bool_v
PolyhedronImplementation<treatInnerT>::InPhiCutoutWedge(
    UnplacedPolyhedron::ZSegment const &segment,
    bool largePhiCutout,
    Vector3D<typename Backend::precision_v> const &point) {
  typedef typename Backend::bool_v Bool_t;
  Bool_t first  = point.Dot(segment.phi.GetNormal(0)) +
                  segment.phi.GetDistance(0) >= 0;
  Bool_t second = point.Dot(segment.phi.GetNormal(1)) +
                  segment.phi.GetDistance(1) < 0;
  // For a cutout larger than 180 degrees, the point is in the wedge if it is
  // in front of at least one plane
  if (largePhiCutout) {
    return first || second;
  }
  // Otherwise it should be in front of both planes
  return first && second;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
bool PolyhedronImplementation<treatInnerT>::ScalarContainsKernel(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &localPoint) {

  // First check if in bounding tube
  {
    bool inBounds;
    HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
        UnplacedContains<kScalar>(
            polyhedron.GetBoundingTube(),
            Vector3D<Precision>(localPoint[0], localPoint[1], localPoint[2]
                                - polyhedron.GetBoundingTubeOffset()),
            inBounds);
    if (!inBounds) return false;
  }

  // Find correct segment by checking Z-bounds
  int zIndex = FindZSegment<kScalar>(polyhedron, localPoint[2]);

  UnplacedPolyhedron::ZSegment const &segment = polyhedron.GetZSegment(zIndex);

  // Check that the point is in the outer shell
  if (!segment.outer.Contains<kScalar>(localPoint)) return false;

  // Check that the point is not in the inner shell
  if (treatInnerT && segment.hasInnerRadius) {
    if (segment.inner.Contains<kScalar>(localPoint)) return false;
  }

  // Check that the point is not in the phi cutout wedge
  if (polyhedron.HasPhiCutout()) {
    return !InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                      localPoint);
  }

  return true;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
Inside_t PolyhedronImplementation<treatInnerT>::ScalarInsideKernel(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &localPoint) {

  // First check if in bounding tube
  {
    bool inBounds;
    HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
        UnplacedContains<kScalar>(
            polyhedron.GetBoundingTube(),
            Vector3D<Precision>(localPoint[0], localPoint[1], localPoint[2]
                                - polyhedron.GetBoundingTubeOffset()),
            inBounds);
    if (!inBounds) return EInside::kOutside;
  }

  // Find correct segment by checking Z-bounds
  int zIndex = FindZSegment<kScalar>(polyhedron, localPoint[2]);

  UnplacedPolyhedron::ZSegment const &segment = polyhedron.GetZSegment(zIndex);

  // Check that the point is in the outer shell
  Inside_t insideTest = segment.outer.Inside<kScalar>(localPoint);
  // Return outside or surface if returned by outer shell
  if (insideTest != EInside::kInside) return insideTest;

  // Check that the point is not in the inner shell
  if (treatInnerT && segment.hasInnerRadius) {
    insideTest = segment.inner.Inside<kScalar>(localPoint);
    if (localPoint == EInside::kInside)  return EInside::kOutside;
    if (localPoint == EInside::kSurface) return EInside::kSurface;
  }

  // Check that the point is not in the phi cutout wedge
  if (polyhedron.HasPhiCutout()) {
    if (InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                  localPoint)) {
      // TODO: check surface of wedge
      return EInside::kOutside;
    }
  }

  return EInside::kInside;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
Precision
PolyhedronImplementation<treatInnerT>::ScalarDistanceToInKernel(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) {

  const Vector3D<Precision> localPoint = transformation.Transform(point);
  const Vector3D<Precision> localDirection =
      transformation.TransformDirection(direction);

  {
    bool inBounds;
    Vector3D<Precision> boundsPoint(
        localPoint[0], localPoint[1],
        localPoint[2]-unplaced.GetBoundingTubeOffset());
    HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
        UnplacedContains<kScalar>(
            unplaced.GetBoundingTube(), boundsPoint, inBounds);
    if (!inBounds) {
      // If the point is outside the bounding tube, check if the ray misses
      // the bounds.
      Precision tubeDistance;
      HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
          DistanceToIn<kScalar>(
              unplaced.GetBoundingTube(), transformation, boundsPoint,
              localDirection, stepMax, tubeDistance);
      if (tubeDistance == kInfinity) return kInfinity;
    }
  }

  int zIndex = FindZSegment<kScalar>(unplaced, localPoint[2]);
  // Don't go out of bounds
  const int zMax = unplaced.GetZSegmentCount();
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = localDirection[2] >= 0;

  Precision distance = kInfinity;
  if (goingRight) {
    for (int zMax = unplaced.GetZSegmentCount(); zIndex < zMax;) {
      distance = DistanceToInZSegment<kScalar>(
          unplaced, zIndex, localPoint, localDirection);
      if (distance >= 0 && distance < kInfinity) break;
      ++zIndex;
      if (unplaced.GetZPlanes()[zIndex] - localPoint[2] > distance) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      distance = DistanceToInZSegment<kScalar>(
          unplaced, zIndex, localPoint, localDirection);
      if (distance >= 0 && distance < kInfinity) break;
      if (localPoint[2] - unplaced.GetZPlanes()[zIndex] > distance) break;
    }
  }

  // Endcaps
  ScalarDistanceToInEndcaps(
      unplaced, goingRight, localPoint, localDirection, distance);

  return distance < stepMax ? distance : stepMax;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
Precision PolyhedronImplementation<treatInnerT>::ScalarSafetyKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point) {

  Precision safety = kInfinity;

  const int zMax = unplaced.GetZSegmentCount();
  int zIndex = FindZSegment<kScalar>(unplaced, point[2]);
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);
  int phiIndex = ScalarFindPhiSegment(unplaced, point);

  // Right
  for (int z = zIndex; z < zMax;) {
    safety = Min(safety, ScalarSafetyToZSegmentSquared(unplaced, z, phiIndex,
                                                       point));
    ++z;
    if (unplaced.GetZPlanes()[z] - point[2] > safety) break;
  }
  // Left
  for (int z = zIndex-1; z >= 0; --z) {
    safety = Min(safety, ScalarSafetyToZSegmentSquared(unplaced, z, phiIndex,
                                                       point));
    if (point[2] - unplaced.GetZPlanes()[z] > safety) break;
  }

  // Endcap
  ScalarSafetyToEndcapsSquared(unplaced, point, safety);

  return sqrt(safety);
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
Precision
PolyhedronImplementation<treatInnerT>::ScalarDistanceToOutKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) {

  int zIndex = FindZSegment<kScalar>(unplaced, point[2]);
  // Don't go out of bounds
  const int zMax = unplaced.GetZSegmentCount();
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = direction[2] >= 0;

  Precision distance = kInfinity;
  if (goingRight) {
    for (int zMax = unplaced.GetZSegmentCount(); zIndex < zMax; ++zIndex) {
      distance = DistanceToOutZSegment<kScalar>(
          unplaced,
          zIndex,
          unplaced.GetZPlane(zIndex),
          unplaced.GetZPlane(zIndex+1),
          point,
          direction);
      if (distance >= 0 && distance < kInfinity) break;
      if (unplaced.GetZPlanes()[zIndex] - point[2] > distance) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      distance = DistanceToOutZSegment<kScalar>(
          unplaced,
          zIndex,
          unplaced.GetZPlane(zIndex),
          unplaced.GetZPlane(zIndex+1),
          point,
          direction);
      if (distance >= 0 && distance < kInfinity) break;
      if (point[2] - unplaced.GetZPlanes()[zIndex] > distance) break;
    }
  }

  // Endcaps
  ScalarDistanceToOutEndcaps(unplaced, goingRight, point, direction, distance);

  return distance < stepMax ? distance : stepMax;
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::UnplacedContains(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Generic UnplacedContains not implemented.\n");
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::Contains(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Generic Contains not implemented.\n");
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "Generic Inside not implemented.\n");
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::DistanceToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "Generic DistanceToIn not implemented.\n");
}

// template <bool treatInnerT>
// template <class Backend>
// VECGEOM_CUDA_HEADER_BOTH
// void PolyhedronImplementation<treatInnerT>::DistanceToIn(
//     UnplacedPolyhedron const &unplaced,
//     Transformation3D const &transformation,
//     Vector3D<typename Backend::precision_v> const &point,
//     Vector3D<typename Backend::precision_v> const &direction,
//     typename Backend::precision_v const &stepMax,
//     typename Backend::precision_v &distance) {

//   typedef typename Backend::precision_v Float_t;
//   typedef typename Backend::bool_v Bool_t;

//   distance = kInfinity;

//   Vector3D<Float_t> localPoint = transformation.Transform(point);
//   Vector3D<Float_t> localDirection = transformation.Transform(direction);

//   // Loop over all Z-planes
//   Float_t distanceTest;
//   for (int i = 0, iMax = unplaced.GetZSegmentCount(); i < iMax; ++i) {
//     UnplacedPolyhedron::Segment const &s = unplaced.GetZSegment(i);
//     distanceTest = s.outer.DistanceToIn<Backend, false>(
//         localPoint, localDirection);
//     MaskedAssign(distanceTest < distance, distanceTest, &distance);
//     if (treatInnerT && s.hasInnerRadius) {
//       distanceTest = s.inner.DistanceToIn<Backend, true>(
//           localPoint, localDirection);
//       MaskedAssign(distanceTest < distance, distanceTest, &distance);
//     }
//   }

//   // Check endcaps
//   Bool_t hitsLeftCap =
//       localPoint[2] < unplaced.GetZPlanes()[0] && localDirection[2] > 0;
//   Bool_t hitsRightCap =
//       localPoint[2] > unplaced.GetZPlanes()[unplaced.GetZSegmentCount()] &&
//       localDirection[2] < 0;
//   if (Any(hitsLeftCap || hitsRightCap)) {
//     distanceTest = kInfinity;
//     MaskedAssign(hitsLeftCap, -localPoint[2] + unplaced.GetZPlanes()[0],
//                  &distanceTest);
//     MaskedAssign(hitsRightCap, -localPoint[2] +
//                  unplaced.GetZPlanes()[unplaced.GetZSegmentCount()],
//                  &distanceTest);
//     distanceTest /= localDirection[2];
//     Vector3D<Float_t> intersection = point + distanceTest*direction;
//     UnplacedPolyhedron::Segment const &leftCap = unplaced.GetZSegment(0);
//     UnplacedPolyhedron::Segment const &rightCap =
//         unplaced.GetZSegment(unplaced.GetZSegmentCount()-1);
//     hitsLeftCap &= leftCap.outer.GetPlanes().Contains<Backend>(intersection);
//     hitsRightCap &= rightCap.outer.GetPlanes().Contains<Backend>(intersection);
//     if (treatInnerT) {
//       if (leftCap.hasInnerRadius) {
//         hitsLeftCap &= !leftCap.inner.GetPlanes().Contains<Backend>(
//             intersection);
//       }
//       if (rightCap.hasInnerRadius) {
//         hitsRightCap &= !rightCap.inner.GetPlanes().Contains<Backend>(
//             intersection);
//       }
//     }
//     MaskedAssign((hitsLeftCap || hitsRightCap) && distanceTest >= 0 &&
//                  distanceTest < distance, distanceTest, &distance);
//   }

//   // Impose upper limit of stepMax on the result
//   MaskedAssign(distance > stepMax && distance < kInfinity, stepMax, &distance);
// }

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::DistanceToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "Generic DistanceToOut not implemented.\n");
}

// template <bool treatInnerT>
// template <class Backend>
// VECGEOM_CUDA_HEADER_BOTH
// void PolyhedronImplementation<treatInnerT>::DistanceToOut(
//     UnplacedPolyhedron const &unplaced,
//     Vector3D<typename Backend::precision_v> const &point,
//     Vector3D<typename Backend::precision_v> const &direction,
//     typename Backend::precision_v const &stepMax,
//     typename Backend::precision_v &distance) {

//   typedef typename Backend::precision_v Float_t;

//   distance = kInfinity;

//   Float_t distanceResult;
//   for (int i = 0, iMax = unplaced.GetZSegmentCount(); i < iMax; ++i) {
//     UnplacedPolyhedron::Segment const &s = unplaced.GetZSegment(i);
//     distanceResult = s.outer.DistanceToOut<Backend>(
//         point, direction, unplaced.GetZPlanes()[i], unplaced.GetZPlanes()[i+1]);
//     MaskedAssign(distanceResult < distance, distanceResult, &distance);
//     if (treatInnerT) {
//       if (s.hasInnerRadius) {
//         distanceResult = s.inner.DistanceToIn<Backend, false>(point, direction);
//         MaskedAssign(distanceResult < distance, distanceResult, &distance);
//       }
//     }
//     // If the next segment is further away than the best distance, no better
//     // distance can be found.
//     if (Backend::early_returns) {
//       if (IsFull(Abs(point[2] - unplaced.GetZPlane(i+1)) > distance)) break;
//     }
//   }
//   distanceResult =
//       unplaced.GetEndCaps().Distance<Backend>(point, direction);
//   MaskedAssign(distanceResult < distance, distanceResult, &distance);

//   MaskedAssign(distance > stepMax && distance < kInfinity, stepMax, &distance);
// }

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "Generic SafetyToIn not implemented.\n");
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "Generic SafetyToOut not implemented.\n");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
