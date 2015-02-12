/// \file PolyhedronImplementation.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include "base/Global.h"

#include "backend/Backend.h"
#ifdef VECGEOM_NVCC
#include "backend/cuda/Backend.h"
#endif
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/kernel/TubeImplementation.h"
#include "volumes/Quadrilaterals.h"
#include "volumes/UnplacedPolyhedron.h"
#include <stdio.h>

namespace vecgeom {

// forward declaration for cuda namespace
// TODO: this is unclear
// we should declare it in a way such that we can use the specialization on the GPU
//
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(PolyhedronImplementation,
              Polyhedron::EInnerRadii, Polyhedron::EInnerRadii::kGeneric,
              Polyhedron::EPhiCutout, Polyhedron::EPhiCutout::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {

   class PlacedPolyhedron;
   class UnplacedPolyhedron;

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
struct PolyhedronImplementation {

    // there is currently no specialization
   static const int transC = translation::kGeneric;
   static const int rotC   = rotation::kGeneric;

   using PlacedShape_t = PlacedPolyhedron;
   using UnplacedShape_t = UnplacedPolyhedron;

   VECGEOM_CUDA_HEADER_BOTH
   static void PrintType() {
     printf("SpecializedPolyhedron<trans = %i, rot = %i, innerR = %i, phicut = %i>", transC, rotC,
            (int)innerRadiiT, (int)phiCutoutT);
   }


  /// \param pointZ Z-coordinate of a point.
  /// \return Index of the Z-segment in which the passed point is located. If
  ///         point is outside the polyhedron, -1 will be returned for Z smaller
  ///         than the first Z-plane, or N for Z larger than the last Z-plane,
  ///         where N is the amount of segments.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::int_v FindZSegment(
      UnplacedPolyhedron const &polyhedron,
      typename Backend::precision_v const &pointZ);

  /// \return Index of the phi-segment in which the passed point is located.
  ///         Assuming the polyhedron has been constructed properly, this should
  ///         always be a valid index.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::int_v FindPhiSegment(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &point);

  /// \param segmentIndex Index to the Z-segment to which the distance should be
  ///                     computed.
  /// \return Distance to the closest quadrilateral intersection by the passed
  ///         ray. Only intersections from the correct direction are accepted,
  ///         so value is always positive.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToInZSegment(
      UnplacedPolyhedron const &polyhedron,
      int segmentIndex,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  /// \param segmentIndex Index to the Z-segment to which the distance should be
  ///                     computed.
  /// \return Distance to the closest quadrilateral intersection by the passed
  ///         ray. Only intersections from the correct direction are accepted,
  ///         so value is always positive.
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

  /// \param segmentIndex Index to the Z-segment for which the safety should be
  ///        computed.
  /// \param phiIndex Index to the phi-segment for which the safety should be
  ///                 computed.
  /// \return Exact squared distance from the passed point to the quadrilateral
  ///         at the Z-segment and phi indices passed.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision ScalarSafetyToZSegmentSquared(
      UnplacedPolyhedron const &polyhedron,
      int segmentIndex,
      int phiIndex,
      Vector3D<Precision> const &point);

  /// \param goingRight Whether the point is travelling along the Z-axis (true)
  ///        or opposite of the Z-axis (false).
  /// \param distance Output argument which will be minimized with the found
  ///                 distance.
  template <bool pointInsideT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ScalarDistanceToEndcaps(
      UnplacedPolyhedron const &polyhedron,
      bool goingRight,
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction,
      Precision &distance);

  /// \brief Computes the exact distance to the closest endcap and minimizes it
  ///        with the output argument.
  /// \param distance Output argument which will be minimized with the found
  ///                 distance.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ScalarSafetyToEndcapsSquared(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &point,
      Precision &distance);

  /// \param largePhiCutout Whether the phi cutout angle is larger than pi.
  /// \return Whether a point is within the infinite phi wedge formed from
  ///         origin in the cutout angle between the first and last vector.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v InPhiCutoutWedge(
      ZSegment const &segment,
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
      // Transformation is passed in order to pass it along as a dummy to the
      // bounding tube's distance function.
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

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
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

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedPolyhedron const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedPolyhedron const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedPolyhedron const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

}; // End struct PolyhedronImplementation

namespace {

/// Polyhedron-specific trait class typedef'ing the tube specialization that
/// should be called as a bounds check in Contains, Inside and DistanceToIn.
template <Polyhedron::EInnerRadii innerRadiiT>
struct HasInnerRadiiTraits {
  /// If polyhedron has inner radii, use a hollow tube
  typedef TubeImplementation<translation::kIdentity,
      rotation::kIdentity, TubeTypes::HollowTube> TubeKernels;
};

template <>
struct HasInnerRadiiTraits<Polyhedron::EInnerRadii::kFalse> {
  /// If polyhedron has no inner radii, use a non-hollow tube
  typedef TubeImplementation<translation::kIdentity,
      rotation::kIdentity, TubeTypes::NonHollowTube> TubeKernels;
};

template <Polyhedron::EInnerRadii innerRadiiT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool TreatInner(bool hasInnerRadius) {
  return hasInnerRadius;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool TreatInner<Polyhedron::EInnerRadii::kFalse>(bool hasInnerRadius) {
  return false;
}

template <Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool TreatPhi(bool hasPhiCutout) {
  return true;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool TreatPhi<Polyhedron::EPhiCutout::kFalse>(bool hasPhiCutout) {
  return false;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool TreatPhi<Polyhedron::EPhiCutout::kGeneric>(bool hasPhiCutout) {
  return hasPhiCutout;
}

template <Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool LargePhiCutout(bool largePhiCutout) {
  return largePhiCutout;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool LargePhiCutout<Polyhedron::EPhiCutout::kTrue>(bool largePhiCutout) {
  return false;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool LargePhiCutout<Polyhedron::EPhiCutout::kLarge>(bool largePhiCutout) {
  return true;
}

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
  // TODO: vectorize this and move the brute-force algorithm to the CUDA
  //       implementation. Inspiration can be found at:
  //       http://schani.wordpress.com/2010/04/30/linear-vs-binary-search/
  int index = -1;
  while (begin < end && pointZ > *begin) {
    ++index;
    ++begin;
  }
  return index;
}

#ifdef VECGEOM_NVCC
template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int FindZSegmentKernel<kCuda>(
    Precision const *begin,
    Precision const *end,
    Precision const &pointZ) {
  // Use scalar version
  return FindZSegmentKernel<kScalar>(begin, end, pointZ);
}
#endif

} // End anonymous namespace

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v
PolyhedronImplementation<innerRadiiT, phiCutoutT>::FindZSegment(
    UnplacedPolyhedron const &polyhedron,
    typename Backend::precision_v const &pointZ) {
  return FindZSegmentKernel<Backend>(
      &polyhedron.GetZPlanes()[0],
      &polyhedron.GetZPlanes()[0]+polyhedron.GetZPlanes().size(),
      pointZ);
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v
PolyhedronImplementation<innerRadiiT, phiCutoutT>::FindPhiSegment(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &point) {

  // Bounds between phi sections are represented as planes through the origin,
  // with the normal pointing along the phi direction.
  // To find the correct section, the point is projected onto each plane. If the
  // point is in front of a plane, but behind the subsequent plane, it must be
  // between them.

  typedef typename Backend::int_v Int_t;
  typedef typename Backend::precision_v Float_t;

  Int_t index(-1);
  SOA3D<Precision> const &phiSections = polyhedron.GetPhiSections();
  Float_t projectionFirst, projectionSecond;
  projectionFirst = point[0]*phiSections.x(0) +
                    point[1]*phiSections.y(0) +
                    point[2]*phiSections.z(0);
  for (int i = 1, iMax = polyhedron.GetSideCount()+1; i < iMax; ++i) {
    projectionSecond = point[0]*phiSections.x(i) +
                       point[1]*phiSections.y(i) +
                       point[2]*phiSections.z(i);
    MaskedAssign(projectionFirst >= 0 && projectionSecond < 0, i-1, &index);
    if (IsFull(index >= 0)) break;
    projectionFirst = projectionSecond;
  }

  return index;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToInZSegment(
    UnplacedPolyhedron const &polyhedron,
    int segmentIndex,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t distance;
  Bool_t done;

  ZSegment const &segment = polyhedron.GetZSegment(segmentIndex);

  // If the outer shell is hit, this will always be the correct result
  distance = segment.outer.DistanceToIn<Backend, false>(point, direction);
  done = distance < kInfinity;
  if (IsFull(done)) return distance;

  // If the outer shell is not hit and the phi cutout sides are hit, this will
  // always be the correct result
  if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout())) {
    MaskedAssign(!done,
                 segment.phi.DistanceToIn<Backend, false>(point, direction),
                 &distance);
  }
  done |= distance < kInfinity;
  if (IsFull(done)) return distance;

  // Finally treat inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius)) {
    MaskedAssign(!done,
                 segment.inner.DistanceToIn<Backend, true>(point, direction),
                 &distance);
  }

  return distance;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToOutZSegment(
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

  ZSegment const &segment = polyhedron.GetZSegment(segmentIndex);

  // Check inner shell first, as it would always be the correct result
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius)) {
    distance = segment.inner.DistanceToIn<Backend, false>(point, direction);
    done = distance < kInfinity;
    if (IsFull(done)) return distance;
  }

  // Check phi cutout if necessary. It is also possible to return here if a
  // result is found
  if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout())) {
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

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision
PolyhedronImplementation<innerRadiiT,
                         phiCutoutT>::ScalarSafetyToZSegmentSquared(
    UnplacedPolyhedron const &polyhedron,
    int segmentIndex,
    int phiIndex,
    Vector3D<Precision> const &point) {

  ZSegment const &segment = polyhedron.GetZSegment(segmentIndex);

  if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout()) &&
      segment.phi.size() == 2 &&
      InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                point)) {
    // If the point is within the phi cutout wedge, the two phi cutout sides are
    // guaranteed to be the closest quadrilaterals to the point.
    return Min(segment.phi.ScalarDistanceSquared(0, point),
               segment.phi.ScalarDistanceSquared(1, point));
  }

  if(phiIndex < 0) return kInfinity;

  // Otherwise check the outer shell
  // TODO: we need to check segment.outer.size() > 0
  Precision safetySquaredOuter = kInfinity;
  if( segment.outer.size() > 0 )
    safetySquaredOuter = segment.outer.ScalarDistanceSquared(phiIndex, point);

  // And finally the inner
  Precision safetySquaredInner = kInfinity;
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius)) {
      if( segment.inner.size() > 0 )
    safetySquaredInner = segment.inner.ScalarDistanceSquared(phiIndex, point);
  }

  return Min(safetySquaredInner, safetySquaredOuter);
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <bool pointInsideT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarDistanceToEndcaps(
    UnplacedPolyhedron const &polyhedron,
    bool goingRight,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    Precision &distance) {

  ZSegment const *segment;
  Precision zPlane;


  // Determine whether to use first segment/first endcap or last segment/second
  // endcap
  // NOTE: might make this more elegant
  if( pointInsideT ) // inside version
  {
      if( direction[2] < 0){
           segment = &polyhedron.GetZSegment(0);
           zPlane = polyhedron.GetZPlane(0);
      }
      else {
          segment = &polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);
          zPlane = polyhedron.GetZPlane(polyhedron.GetZSegmentCount());
      }
  }
  else // outside version
  {
      if( direction[2] < 0){
          segment = &polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);
          zPlane = polyhedron.GetZPlane(polyhedron.GetZSegmentCount());
      }
      else{
          segment = &polyhedron.GetZSegment(0);
          zPlane = polyhedron.GetZPlane(0);
      }
  }

  // original formulation had a bug:
//  if (Flip<pointInsideT>::FlipLogical(goingRight) &&
//      point[2] > Flip<!pointInsideT>::FlipSign(polyhedron.GetZPlane(0))) {
//
//      segment = &polyhedron.GetZSegment(0);
//      zPlane = polyhedron.GetZPlane(0);
//
//  }
//  else if (Flip<!pointInsideT>::FlipLogical(goingRight) &&
//             point[2] > Flip<pointInsideT>::FlipSign(
//                 polyhedron.GetZPlane(polyhedron.GetZSegmentCount()))) {
//
//      segment = &polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);
//      zPlane = polyhedron.GetZPlane(polyhedron.GetZSegmentCount());
//  } else {
//    return;
//  }

  Precision distanceTest = (zPlane - point[2]) / direction[2];
  // If the distance is not better there's no reason to check for validity
  if (distanceTest < 0 || distanceTest >= distance) return;

  Vector3D<Precision> intersection = point + distanceTest*direction;
  // Intersection point must be inside outer shell and outside inner shell
  if (!segment->outer.Contains<kScalar>(intersection)) return;
  if (TreatInner<innerRadiiT>(segment->hasInnerRadius)) {
    if (segment->inner.Contains<kScalar>(intersection)) return;
  }
  // Intersection point must not be in phi cutout wedge
  if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout())) {
    if (InPhiCutoutWedge<kScalar>(*segment, polyhedron.HasLargePhiCutout(),
                                  intersection)) {
      return;
    }
  }

  distance = distanceTest;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void
PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyToEndcapsSquared(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &point,
    Precision &distanceSquared) {

  // Compute both distances (simple subtractions) to determine which is closer
  Precision firstDistance = polyhedron.GetZPlane(0) - point[2];
  Precision lastDistance =
      polyhedron.GetZPlane(polyhedron.GetZSegmentCount()) - point[2];

  // Only treat the closest endcap
  bool isFirst = Abs(firstDistance) < Abs(lastDistance);
  ZSegment const &segment =
      isFirst ? polyhedron.GetZSegment(0)
              : polyhedron.GetZSegment(polyhedron.GetZSegmentCount()-1);

  Precision distanceTest = isFirst ? firstDistance : lastDistance;
  Precision distanceTestSquared = distanceTest*distanceTest;
  // No need to investigate further if distance is larger anyway
  if (distanceTestSquared >= distanceSquared) return;

  // Check if projection is within the endcap bounds
  Vector3D<Precision> intersection(point[0], point[1], point[2] + distanceTest);
  if (!segment.outer.Contains<kScalar>(intersection)) return;
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius)) {
    if (segment.inner.Contains<kScalar>(intersection)) return;
  }
  if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout())) {
    if (InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                  intersection)) {
      return;
    }
  }

  distanceSquared = distanceTestSquared;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::bool_v
PolyhedronImplementation<innerRadiiT, phiCutoutT>::InPhiCutoutWedge(
    ZSegment const &segment,
    bool largePhiCutout,
    Vector3D<typename Backend::precision_v> const &point) {
  typedef typename Backend::bool_v Bool_t;
  Bool_t first  = point.Dot(segment.phi.GetNormal(0)) +
                  segment.phi.GetDistance(0) >= 0;
  Bool_t second = point.Dot(segment.phi.GetNormal(1)) +
                  segment.phi.GetDistance(1) >= 0;
  // For a cutout larger than 180 degrees, the point is in the wedge if it is
  // in front of at least one plane.
  if (LargePhiCutout<phiCutoutT>(largePhiCutout)) {
    return first || second;
  }
  // Otherwise it should be in front of both planes
  return first && second;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
bool PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarContainsKernel(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &localPoint) {

  // First check if in bounding tube
  {
    bool inBounds;
    // Correct tube algorithm obtained from trait class
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template
        UnplacedContains<kScalar>(
            polyhedron.GetBoundingTube(),
            Vector3D<Precision>(localPoint[0], localPoint[1], localPoint[2]
                                - polyhedron.GetBoundingTubeOffset()),
            inBounds);
    if (!inBounds) return false;
  }

  // Find correct segment by checking Z-bounds
  int zIndex = FindZSegment<kScalar>(polyhedron, localPoint[2]);
  if( ! (zIndex >= 0) ) return false;

  ZSegment const &segment = polyhedron.GetZSegment(zIndex);

  // Check that the point is in the outer shell
  if (!segment.outer.Contains<kScalar>(localPoint)) return false;

  // Check that the point is not in the inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius)) {
    if (segment.inner.Contains<kScalar>(localPoint)) return false;
  }

  // Check that the point is not in the phi cutout wedge
  // NOTE: This check is already part of the bounding tube check
  // this code should be removed
  //if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout())) {
  //  return !InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                     //localPoint);
  //}

  return true;
}

// TODO: check this code -- maybe unify with previous function
template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Inside_t PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarInsideKernel(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &localPoint) {

  // First check if in bounding tube
  {
    bool inBounds;
    // Correct tube algorithm obtained from trait class
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template
        UnplacedContains<kScalar>(
            polyhedron.GetBoundingTube(),
            Vector3D<Precision>(localPoint[0], localPoint[1], localPoint[2]
                                - polyhedron.GetBoundingTubeOffset()),
            inBounds);
    if (!inBounds) return EInside::kOutside;
  }

  // Find correct segment by checking Z-bounds
  int zIndex = FindZSegment<kScalar>(polyhedron, localPoint[2]);

  ZSegment const &segment = polyhedron.GetZSegment(zIndex);

  // Check that the point is in the outer shell
  {
    Inside_t insideOuter = segment.outer.Inside<kScalar>(localPoint);
    if (insideOuter != EInside::kInside) return insideOuter;
  }

  // Check that the point is not in the inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius)) {
    Inside_t insideInner = segment.inner.Inside<kScalar>(localPoint);
    if (insideInner == EInside::kInside)  return EInside::kOutside;
    if (insideInner == EInside::kSurface) return EInside::kSurface;
  }

  // Check that the point is not in the phi cutout wedge
  if (TreatPhi<phiCutoutT>(polyhedron.HasPhiCutout())) {
    if (InPhiCutoutWedge<kScalar>(segment, polyhedron.HasLargePhiCutout(),
                                  localPoint)) {
      // TODO: check for surface case when in phi wedge. This can be done by
      //       checking the distance to planes of the phi cutout sides.
      return EInside::kOutside;
    }
  }

  return EInside::kInside;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision
PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarDistanceToInKernel(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) {

  // Transformation is done here so the transformation can be used as a dummy
  // argument to the bounding tube's distance function.
  const Vector3D<Precision> localPoint = transformation.Transform(point);
  const Vector3D<Precision> localDirection =
      transformation.TransformDirection(direction);


  {
    // Check if the point is within the bounding tube
    bool inBounds;
    Vector3D<Precision> boundsPoint(
        localPoint[0], localPoint[1],
        localPoint[2]-unplaced.GetBoundingTubeOffset());
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template
        UnplacedContains<kScalar>(
            unplaced.GetBoundingTube(), boundsPoint, inBounds);
    // If the point is inside the bounding tube, the result of DistanceToIn is
    // unreliable and cannot be used to reject rays.
    // TODO: adjust tube DistanceToIn function to correctly return a negative
    //       value for points inside the tube. This will allow the removal of
    //       the contains check here.
    if (!inBounds) {
      // If the point is outside the bounding tube, check if the ray misses
      // the bounds
      Precision tubeDistance;
      HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template
          DistanceToIn<kScalar>(
              unplaced.GetBoundingTube(), transformation, boundsPoint,
              localDirection, stepMax, tubeDistance);
      if (tubeDistance == kInfinity)
          {
            return kInfinity;
          }
    }
  }

  int zIndex = FindZSegment<kScalar>(unplaced, localPoint[2]);
  const int zMax = unplaced.GetZSegmentCount();
  // Don't go out of bounds here, as the first/last segment should be checked
  // even if the point is outside of Z-bounds
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = localDirection[2] >= 0;

  Precision distance = kInfinity;
  if (goingRight) {
    for (int zMax = unplaced.GetZSegmentCount(); zIndex < zMax; ++zIndex) {
      distance = DistanceToInZSegment<kScalar>(
          unplaced, zIndex, localPoint, localDirection);
      // No segment further away can be at a shorter distance to the point, so
      // if a valid distance is found, only endcaps remain to be investigated
      if (distance >= 0 && distance < kInfinity) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      distance = DistanceToInZSegment<kScalar>(
          unplaced, zIndex, localPoint, localDirection);
      // No segment further away can be at a shorter distance to the point, so
      // if a valid distance is found, only endcaps remain to be investigated
      if (distance >= 0 && distance < kInfinity) break;
    }
  }

  // Minimize with distance to endcaps
  ScalarDistanceToEndcaps<false>(
      unplaced, goingRight, localPoint, localDirection, distance);

  // Don't exceed stepMax
  return distance < stepMax ? distance : stepMax;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point) {

  Precision safety = kInfinity;

  const int zMax = unplaced.GetZSegmentCount();
  int zIndex = FindZSegment<kScalar>(unplaced, point[2]);
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);
  int phiIndex = FindPhiSegment<kScalar>(unplaced, point);

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

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision
PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarDistanceToOutKernel(
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
  ScalarDistanceToEndcaps<true>(unplaced, goingRight, point, direction,
                                distance);

  return distance < stepMax ? distance : stepMax;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::UnplacedContains(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

    inside = ScalarContainsKernel( unplaced, localPoint );
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::Contains(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {
 // Assert(0, "Generic Contains not implemented.\n");

 // we should assert if Backend != scalar
    localPoint = transformation.Transform<transC,rotC>(point);
    inside = ScalarContainsKernel( unplaced, localPoint);
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

    // we should assert if Backend != scalar
    inside = ScalarInsideKernel( unplaced, transformation.Transform<transC,rotC>(point));
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
 // Assert(0, "Generic DistanceToIn not implemented.\n");
    distance = ScalarDistanceToInKernel( unplaced, transformation,
           point, direction, stepMax) ;
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    distance = ScalarDistanceToOutKernel( unplaced, point, direction, stepMax);
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

     safety = ScalarSafetyKernel(
           unplaced, transformation.Transform<transC,rotC>(point) );
}

template <Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

    safety = ScalarSafetyKernel(unplaced, point);
}

} // End inline namespace
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
