/// \file PolyhedronImplementation.h
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

  /// \param pointZ Z-coordinate of a point.
  /// \return Index of the Z-segment in which the passed point is located. If
  ///         point is outside the polyhedron, -1 will be returned for Z smaller
  ///         than the first Z-plane, or N for Z larger than the last Z-plane,
  ///         where N is the amount of segments.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static int FindZSegment(UnplacedStruct_t const &unplaced,
                                                                       Real_v const &pointZ);

  /// \return Index of the phi-segment in which the passed point is located.
  ///         Assuming the polyhedron has been constructed properly, this should
  ///         always be a valid index.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static int FindPhiSegment(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point);

  /// \param segmentIndex Index to the Z-segment to which the distance should be
  ///                     computed.
  /// \return Distance to the closest quadrilateral intersection by the passed
  ///         ray. Only intersections from the correct direction are accepted,
  ///         so value is always positive.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v DistanceToInZSegment(UnplacedStruct_t const &unplaced,
                                                                                  int segmentIndex,
                                                                                  Vector3D<Real_v> const &point,
                                                                                  Vector3D<Real_v> const &direction);

  /// \param segmentIndex Index to the Z-segment to which the distance should be
  ///                     computed.
  /// \return Distance to the closest quadrilateral intersection by the passed
  ///         ray. Only intersections from the correct direction are accepted,
  ///         so value is always positive.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v DistanceToOutZSegment(UnplacedStruct_t const &unplaced,
                                                                                   int segmentIndex, Precision zMin,
                                                                                   Precision zMax,
                                                                                   Vector3D<Real_v> const &point,
                                                                                   Vector3D<Real_v> const &direction);

  /// \param segmentIndex Index to the Z-segment for which the safety should be
  ///        computed.
  /// \param phiIndex Index to the phi-segment for which the safety should be
  ///                 computed.
  /// \return Exact squared distance from the passed point to the quadrilateral
  ///         at the Z-segment and phi indices passed.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision ScalarSafetyToZSegmentSquared(UnplacedStruct_t const &unplaced, int segmentIndex, int &phiIndex,
                                                 Vector3D<Precision> const &point, bool pt_inside, int &iSurf);

  /// \param goingRight Whether the point is travelling along the Z-axis (true)
  ///        or opposite of the Z-axis (false).
  /// \param distance Output argument which will be minimized with the found
  ///                 distance.
  template <bool pointInsideT>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ScalarDistanceToEndcaps(UnplacedStruct_t const &unplaced,
                                                                                   bool goingRight,
                                                                                   Vector3D<Precision> const &point,
                                                                                   Vector3D<Precision> const &direction,
                                                                                   Precision &distance);

  /// \brief Computes the exact distance to the closest endcap and minimizes it
  ///        with the output argument.
  /// \param distance Output argument which will be minimized with the found
  ///                 distance.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static void ScalarSafetyToEndcapsSquared(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                           Precision &distance, int &iz);

  /// \param largePhiCutout Whether the phi cutout angle is larger than pi.
  /// \return Whether a point is within the infinite phi wedge formed from
  ///         origin in the cutout angle between the first and last vector.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static vecCore::Mask_v<Real_v> InPhiCutoutWedge(
      ZSegment const &segment, bool largePhiCutout, Vector3D<Real_v> const &point);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool ScalarContainsKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Inside_t ScalarInsideKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Inside_t ScalarInsideSegPhi(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex,
                                     int phiIndex);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Inside_t ScalarInsideSegBorder(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision ScalarDistanceToInKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                            Vector3D<Precision> const &direction, const Precision stepMax);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision ScalarDistanceToOutKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                             Vector3D<Precision> const &direction, const Precision stepMax);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision ScalarSafetyKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                      bool pt_inside);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool ScalarNormalKernel(UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point,
                                 Vector3D<Precision> &normal);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContains(UnplacedStruct_t const &unplaced,
                                                                            Vector3D<Real_v> const &point,
                                                                            Bool_v &inside);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside);

  /// Not implemented. Scalar version is called from Specializedunplaced.
  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety);

  /// Not implemented. Scalar version is called from SpecializedPolyhedron.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety);

}; // End struct PolyhedronImplementation

namespace {

/// Polyhedron-specific trait class typedef'ing the tube specialization that
/// should be called as a bounds check in Contains, Inside and DistanceToIn.

// SW (19.6.2015): switching to UniversalTube as Phi section was not
// correctly treated with a hollow tube
// TODO: this could be CORRECTLY put back for optimization
template <Polyhedron::EInnerRadii innerRadiiT>
struct HasInnerRadiiTraits {
  /// If polyhedron has inner radii, use a hollow tube
  typedef TubeImplementation<TubeTypes::UniversalTube> TubeKernels;
};

template <>
struct HasInnerRadiiTraits<Polyhedron::EInnerRadii::kFalse> {
  /// If polyhedron has no inner radii, use a non-hollow tube
  typedef TubeImplementation<TubeTypes::UniversalTube> TubeKernels;
};

template <Polyhedron::EInnerRadii innerRadiiT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatInner(bool hasInnerRadius)
{
  return hasInnerRadius;
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatInner<Polyhedron::EInnerRadii::kFalse>(bool /*hasInnerRadius*/)
{
  return false;
}

template <Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatPhi(bool /*hasPhiCutout*/)
{
  return true;
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatPhi<Polyhedron::EPhiCutout::kFalse>(bool /*hasPhiCutout*/)
{
  return false;
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool TreatPhi<Polyhedron::EPhiCutout::kGeneric>(bool hasPhiCutout)
{
  return hasPhiCutout;
}

template <Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool LargePhiCutout(bool largePhiCutout)
{
  return largePhiCutout;
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool LargePhiCutout<Polyhedron::EPhiCutout::kTrue>(bool /*largePhiCutout*/)
{
  return false;
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool LargePhiCutout<Polyhedron::EPhiCutout::kLarge>(
    bool /*largePhiCutout*/)
{
  return true;
}

} // End anonymous namespace

namespace {

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
  if ((index + 2 < size) && (pointZ > *(end + 1) - kTolerance)) return (index + 1);
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
    vecCore__MaskedAssignFunc(index, projectionFirst > -kTolerance && projectionSecond < kTolerance, i - 1);
    if (vecCore::MaskFull(index >= 0)) break;
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

  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v distance;
  Bool_v done;

  ZSegment const &segment = unplaced.fZSegments[segmentIndex];

  // If the outer shell is hit, this will always be the correct result
  distance = segment.outer.DistanceToIn<Real_v, false>(point, direction);
  done     = distance < InfinityLength<Real_v>();
  if (vecCore::MaskFull(done)) return distance;

  // If the outer shell is not hit and the phi cutout sides are hit, this will
  // always be the correct result
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    vecCore__MaskedAssignFunc(distance, !done, (segment.phi.DistanceToIn<Real_v, false>(point, direction)));
    if (unplaced.fHasLargePhiCutout) {
      // NOTE: The statement above is NOT always true: if fHasLargePhiCutout is false there can be a first hit of the
      // inner surface coming from the endcap holes
      done |= distance < InfinityLength<Real_v>();
      if (vecCore::MaskFull(done)) return distance;
    }
  }

  // Finally treat inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    Real_v distrmin = segment.inner.DistanceToIn<Real_v, true>(point, direction);
    vecCore__MaskedAssignFunc(distance, !done && distance > distrmin, distrmin);
  }

  return distance;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToOutZSegment(
    UnplacedStruct_t const &unplaced, int segmentIndex, Precision zMin, Precision zMax, Vector3D<Real_v> const &point,
    Vector3D<Real_v> const &direction)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  Bool_v done(false);
  Real_v distance = InfinityLength<Real_v>();

  ZSegment const &segment = unplaced.fZSegments[segmentIndex];

  // Check inner shell first, as it would always be the correct result
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    distance = segment.inner.DistanceToIn<Real_v, false>(point, direction);
    // Even if an inner surface is hit, there may be a phi hit before if there is no large phi cut
    if (unplaced.fHasLargePhiCutout) {
      done = distance < InfinityLength<Real_v>();
      if (vecCore::MaskFull(done)) return distance;
    }
  }

  // Check phi cutout if necessary. It is also possible to return here if a
  // result is found
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    Real_v distphi = segment.phi.DistanceToIn<Real_v, true>(point, direction);
    vecCore::MaskedAssign(distance, distance > distphi, distphi);
  }

  done = distance > -kTolerance && distance < InfinityLength<Real_v>();
  if (vecCore::MaskFull(done)) return distance;

  // Finally check outer shell
  Real_v distout = segment.outer.DistanceToOut<Real_v>(point, direction, zMin, zMax);
  vecCore::MaskedAssign(distance, !done, distout);

  return distance;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Precision
PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyToZSegmentSquared(UnplacedStruct_t const &unplaced,
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
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<
    innerRadiiT, phiCutoutT>::ScalarDistanceToEndcaps(UnplacedStruct_t const &unplaced, bool /*goingRight*/,
                                                      Vector3D<Precision> const &point,
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
    if (InPhiCutoutWedge<Precision>(*segment, unplaced.fHasLargePhiCutout, intersection)) {
      return;
    }
  }

  distance = distanceTest;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<
    innerRadiiT, phiCutoutT>::ScalarSafetyToEndcapsSquared(UnplacedStruct_t const &unplaced,
                                                           Vector3D<Precision> const &point, Precision &distanceSquared,
                                                           int &iz)
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
    if (InPhiCutoutWedge<Precision>(segment, unplaced.fHasLargePhiCutout, intersection)) {
      return;
    }
  }

  iz              = (isFirst) ? -1 : 1;
  distanceSquared = distanceTestSquared;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecCore::Mask_v<Real_v> PolyhedronImplementation<
    innerRadiiT, phiCutoutT>::InPhiCutoutWedge(ZSegment const &segment, bool largePhiCutout,
                                               Vector3D<Real_v> const &point)
{
  using Bool_v     = vecCore::Mask_v<Real_v>;
  Bool_v pointSeg0 = point.Dot(segment.phi.GetNormal(0)) + segment.phi.GetDistance(0) >= 0;
  Bool_v pointSeg1 = point.Dot(segment.phi.GetNormal(1)) + segment.phi.GetDistance(1) >= 0;
  // For a cutout larger than 180 degrees, the point is in the wedge if it is
  // in front of at least one plane.
  if (LargePhiCutout<phiCutoutT>(largePhiCutout)) {
    return pointSeg0 || pointSeg1;
  }
  // Otherwise it should be in front of both planes
  return pointSeg0 && pointSeg1;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE bool PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarContainsKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point)
{

  // First check if in bounding tube
  {
    bool inBounds;
    // Correct tube algorithm obtained from trait class
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template Contains<>(
        unplaced.fBoundingTube, Vector3D<Precision>(point[0], point[1], point[2] - unplaced.fBoundingTubeOffset),
        inBounds);
    if (!inBounds) return false;
  }

  // Find correct segment by checking Z-bounds
  int zIndex = FindZSegment<Precision>(unplaced, point[2]);
  if (!((zIndex >= 0) && (zIndex < unplaced.fZSegments.size()))) return false;

  ZSegment const &segment = unplaced.fZSegments[zIndex];

  // In case the point lies at the same Z as 2 consecutive planes, the lesser
  // index is selected. The Quadrilaterals algorithm for Contains in this case
  // does not work.
  if (unplaced.fSameZ[zIndex]) {
    // Identify phi index
    int phiIndex = FindPhiSegment<Precision>(unplaced, point);
    if (phiIndex < 0) return false;
    // Get the vector perpendicular to the rmax edge of the outer quadrilateral
    Vector3D<Precision> const &vout = (segment.outer.size()) ? segment.outer.GetSideVectors()[0].GetNormals()[phiIndex]
                                                             : segment.inner.GetSideVectors()[0].GetNormals()[phiIndex];
    // Compute the projection of the point vectoron the vout vector. This
    // corresponds to a "radius" or the point.
    Precision rdotvout = vecCore::math::Abs<Precision>(point.Dot(vout));
    // Now compare the point radius with the ranges corresponding to the lower
    // and upper segments
    bool in1 = (rdotvout >= unplaced.fRMin[zIndex]) && (rdotvout <= unplaced.fRMax[zIndex]);
    bool in2 = (rdotvout >= unplaced.fRMin[zIndex + 1]) && (rdotvout <= unplaced.fRMax[zIndex + 1]);
    return (in1 | in2);
  }

  // Check that the point is in the outer shell
  if (!segment.outer.Contains<Precision>(point)) return false;

  // Check that the point is not in the inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    if (segment.inner.Contains<Precision>(point)) return false;
  }

  // In principle, handling of phi should not be needed here since it is
  // contained in the bounding tube. However, we need to check again due
  // to different handling of tolerances.
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    if (!segment.phi.Contains<Precision>(point)) return false;
  }

  return true;
}

// TODO: check this code -- maybe unify with previous function
template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE Inside_t PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarInsideKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point)
{

  // First check if in bounding tube
  {
    bool inBounds;
    // Correct tube algorithm obtained from trait class
    // FIX: the bounding tube was wrong. Since the fast UnplacedContains is
    // used for early return, the bounding tube has to be larger than the
    // ideal bounding tube to account for the tolerance (offset was wrong)
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template Contains<>(
        unplaced.fBoundingTube, Vector3D<Precision>(point[0], point[1], point[2] - unplaced.fBoundingTubeOffset),
        inBounds);
    if (!inBounds) return EInside::kOutside;
  }

  // Find correct segment by checking Z-bounds
  // The FindZSegment was fixed for the degenerated Z case when 2 planes
  // have identical Z. In this case, if the point is close within tolerance
  // to such section, the returned index has to be the first of the 2, so that
  // all navigation functions start by checking the degenerated segment.
  int zIndex = FindZSegment<Precision>(unplaced, point[2]);
  if (zIndex > (unplaced.fZSegments.size() - 1)) zIndex = unplaced.fZSegments.size() - 1;
  if (zIndex < 0) zIndex = 0;

  ZSegment const &segment = unplaced.fZSegments[zIndex];

  // Point in between 2 planes at same Z
  if (unplaced.fSameZ[zIndex]) return ScalarInsideSegBorder(unplaced, point, zIndex);

  // Check that the point is in the outer shell
  {
    Inside_t insideOuter = segment.outer.Inside<Precision, Inside_t>(point);
    if (insideOuter != EInside::kInside) return insideOuter;
  }

  // Check that the point is not in the inner shell
  if (TreatInner<innerRadiiT>(segment.hasInnerRadius())) {
    Inside_t insideInner = segment.inner.Inside<Precision, Inside_t>(point);
    if (insideInner == EInside::kInside) return EInside::kOutside;
    if (insideInner == EInside::kSurface) return EInside::kSurface;
  }

  // Check that the point is not in the phi cutout wedge
  if (TreatPhi<phiCutoutT>(unplaced.fHasPhiCutout)) {
    // Inside_t insidePhi = unplaced.fPhiWedge.Inside<Precision, Inside_t>(point);
    Inside_t insidePhi = segment.phi.Inside<Precision, Inside_t>(point);
    if (insidePhi != EInside::kInside) return insidePhi;
  }

  // FIX: Still need to check if not on one of the Z boundaries.
  Precision dz = vecCore::math::Abs(vecCore::math::Abs(point[2] - unplaced.fBoundingTubeOffset) -
                                    0.5 * (unplaced.fZPlanes[unplaced.fZSegments.size()] - unplaced.fZPlanes[0]));
  if (dz < kTolerance) return EInside::kSurface;
  return EInside::kInside;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE Inside_t PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarInsideSegBorder(
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
VECCORE_ATT_HOST_DEVICE Inside_t PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarInsideSegPhi(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, int zIndex, int phiIndex)
{
  // Check inside for a specified z segment and phi edge
  if (phiIndex < 0) return EInside::kOutside;

  // Z range
  Precision dz = vecCore::math::Abs(point[2] - unplaced.fBoundingTubeOffset) -
                 0.5 * (unplaced.fZPlanes[unplaced.fZSegments.size()] - unplaced.fZPlanes[0]);
  //  if (vecCore::math::Abs(dz) < kHalfTolerance) return EInside::kSurface;
  if (dz > kHalfTolerance) return EInside::kOutside;

  if (unplaced.fSameZ[zIndex]) return ScalarInsideSegBorder(unplaced, point, zIndex);

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
VECCORE_ATT_HOST_DEVICE Precision PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarDistanceToInKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
    const Precision stepMax)
{

  // Fast exclude points beyond endcaps moving on same side as endcap normal
  if ((point[2] < unplaced.fZPlanes[0] + kTolerance) && direction[2] <= 0) return InfinityLength<Precision>();
  if ((point[2] > unplaced.fZPlanes[unplaced.fZSegments.size()] - kTolerance) && direction[2] >= 0)
    return InfinityLength<Precision>();

  // Perform explicit Inside check to detect wrong side points. This impacts
  // DistanceToIn performance by about 5% for all topologies
  auto inside = ScalarInsideKernel(unplaced, point);
  if (inside == kInside) return -1.;

  // Check if the point is within the bounding tube
  bool inBounds;
  Precision tubeDistance = 0.;
  {
    Vector3D<Precision> boundsPoint(point[0], point[1], point[2] - unplaced.fBoundingTubeOffset);
    HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template Contains<>(unplaced.fBoundingTube, boundsPoint, inBounds);
    // If the point is inside the bounding tube, the result of DistanceToIn is
    // unreliable and cannot be used to reject rays.
    // TODO: adjust tube DistanceToIn function to correctly return a negative
    //       value for points inside the tube. This will allow the removal of
    //       the contains check here.
    if (!inBounds) {
      // If the point is outside the bounding tube, check if the ray misses
      // the bounds
      HasInnerRadiiTraits<innerRadiiT>::TubeKernels::template DistanceToIn<>(unplaced.fBoundingTube, boundsPoint,
                                                                             direction, stepMax, tubeDistance);
      if (tubeDistance == InfinityLength<Precision>()) {
        return InfinityLength<Precision>();
      }
    }
  }

  int zIndex     = FindZSegment<Precision>(unplaced, point[2]);
  const int zMax = unplaced.fZSegments.size();
  // Don't go out of bounds here, as the first/last segment should be checked
  // even if the point is outside of Z-bounds
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = direction[2] >= 0;

  Precision distance = InfinityLength<Precision>();
  if (goingRight) {
    for (int zSegCount = unplaced.fZSegments.size(); zIndex < zSegCount; ++zIndex) {
      distance = DistanceToInZSegment<Precision>(unplaced, zIndex, point, direction);
      // No segment further away can be at a shorter distance to the point, so
      // if a valid distance is found, only endcaps remain to be investigated
      if (distance >= 0 && distance < InfinityLength<Precision>()) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      distance = DistanceToInZSegment<Precision>(unplaced, zIndex, point, direction);
      // No segment further away can be at a shorter distance to the point, so
      // if a valid distance is found, only endcaps remain to be investigated
      if (distance >= 0 && distance < InfinityLength<Precision>()) break;
    }
  }

  // Minimize with distance to endcaps
  ScalarDistanceToEndcaps<false>(unplaced, goingRight, point, direction, distance);

  // last sanity check: distance should be larger than estimate from bounding tube
  return (distance >= tubeDistance - 1E-6) ? distance : vecgeom::InfinityLength<Precision>();
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE Precision PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, bool pt_inside)
{

  Precision safety = InfinityLength<Precision>();
  Precision dz;
  int iSurf, iz;

  const int zMax = unplaced.fZSegments.size();
  int zIndex     = FindZSegment<Precision>(unplaced, point[2]);
  zIndex         = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  int phiIndex = FindPhiSegment<Precision>(unplaced, point);

  // Check if point is on the 'pt_inside' side
  // Perform explicit Inside check to detect wrong side points. This impacts
  // Safety performance by 5-10% for all topologies
  Inside_t inside = ScalarInsideSegPhi(unplaced, point, zIndex, phiIndex);
  if (inside == EInside::kSurface) return 0.;
  bool contains = (inside == EInside::kInside);
  if (contains ^ pt_inside) return -1.;

  // Right
  for (int z = zIndex; z < zMax;) {
    safety = Min(safety, ScalarSafetyToZSegmentSquared(unplaced, z, phiIndex, point, pt_inside, iSurf));
    ++z;
    dz = unplaced.fZPlanes[z] - point[2];
    // Fixed bug: dz was compared directly to safety to stop the search, while safety is a squared
    if (dz * dz > safety) break;
  }
  // Left
  for (int z = zIndex - 1; z >= 0; --z) {
    safety = Min(safety, ScalarSafetyToZSegmentSquared(unplaced, z, phiIndex, point, pt_inside, iSurf));
    dz     = point[2] - unplaced.fZPlanes[z];
    if (dz * dz > safety) break;
  }

  // Endcap
  ScalarSafetyToEndcapsSquared(unplaced, point, safety, iz);

  safety = vecCore::math::Sqrt(safety);
  return safety;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
VECCORE_ATT_HOST_DEVICE bool PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarNormalKernel(
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
    Precision safetySeg = ScalarSafetyToZSegmentSquared(unplaced, z, phiIndex, point, true, iSurfCrt);
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
    Precision safetySeg = ScalarSafetyToZSegmentSquared(unplaced, z, phiIndex, point, true, iSurfCrt);
    if (safetySeg < safety) {
      safety = safetySeg;
      iSeg   = z;
      iSurf  = iSurfCrt;
    }
    dz = point[2] - unplaced.fZPlanes[z];
    if (dz * dz > safety) break;
  }

  // Endcap
  ScalarSafetyToEndcapsSquared(unplaced, point, safety, iz);
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
VECCORE_ATT_HOST_DEVICE Precision PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarDistanceToOutKernel(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
    const Precision /*stepMax*/)
{
  // Fast exclusion if out of Z range
  const int zMax = unplaced.fZSegments.size();
  if ((point[2] < unplaced.fZPlanes[0] - kTolerance) || (point[2] > unplaced.fZPlanes[zMax] + kTolerance)) return -1.;

  // Perform explicit Inside check to detect wrong side points. This impacts
  // DistanceToOut performance by about 20% for all topologies
  auto inside = ScalarInsideKernel(unplaced, point);
  if (inside == kOutside) return -1.;

  int zIndex = FindZSegment<Precision>(unplaced, point[2]);
  // Don't go out of bounds
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = direction[2] >= 0;

  Precision distance = InfinityLength<Precision>();
  if (goingRight) {
    for (; zIndex < zMax; ++zIndex) {
      distance = DistanceToOutZSegment<Precision>(unplaced, zIndex, unplaced.fZPlanes[zIndex],
                                                  unplaced.fZPlanes[zIndex + 1], point, direction);
      if (distance >= 0 && distance < InfinityLength<Precision>()) break;
      if (unplaced.fZPlanes[zIndex] - point[2] > distance) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      distance = DistanceToOutZSegment<Precision>(unplaced, zIndex, unplaced.fZPlanes[zIndex],
                                                  unplaced.fZPlanes[zIndex + 1], point, direction);
      if (distance >= 0 && distance < InfinityLength<Precision>()) break;
      if (point[2] - unplaced.fZPlanes[zIndex] > distance) break;
    }
  }

  // Endcaps
  ScalarDistanceToEndcaps<true>(unplaced, goingRight, point, direction, distance);

  // disabling stepMax until convention revised and clear
  // there is a problem when distance = infinity due to some error condition but stepMax finite
  // return distance < stepMax ? distance : stepMax;
  // If not hitting anything, we must be on an edge since point is not outside
  if (distance >= InfinityLength<Precision>()) distance = 0.;
  return distance;
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v, typename Bool_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::UnplacedContains(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
{

  inside = ScalarContainsKernel(unplaced, point);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v, typename Bool_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::Contains(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
{

  // we should assert if Backend != scalar
  inside = ScalarContainsKernel(unplaced, point);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v, typename Inside_t>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::Inside(UnplacedStruct_t const &unplaced,
                                                                                       Vector3D<Real_v> const &point,
                                                                                       Inside_t &inside)
{

  // we should assert if Backend != scalar
  inside = ScalarInsideKernel(unplaced, point);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToIn(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{
  distance = ScalarDistanceToInKernel(unplaced, point, direction, stepMax);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::DistanceToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{

  distance = ScalarDistanceToOutKernel(unplaced, point, direction, stepMax);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToIn(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{

  safety = ScalarSafetyKernel(unplaced, point, false);
}

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PolyhedronImplementation<innerRadiiT, phiCutoutT>::SafetyToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{

  safety = ScalarSafetyKernel(unplaced, point, true);
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
