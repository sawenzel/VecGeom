/// \file PolyhedronImplementation.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include "base/Global.h"

#include "backend/Backend.h"
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
  static typename Backend::int_v FindZSegment(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::int_v FindPhiSegment(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInZSegment(
    UnplacedPolyhedron::Segment const &segment,
    int sideCount,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToOutZSegment(
    UnplacedPolyhedron::Segment const &segment,
    int sideCount,
    Precision zMin,
    Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction);

  template <bool treatSurfaceT>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ScalarInsideKernel(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &localPoint,
      typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t &inside);

  VECGEOM_CUDA_HEADER_BOTH
  static inline Precision ScalarDistanceToInKernel(
      UnplacedPolyhedron const &unplaced,
      Vector3D<Precision> const &localPoint,
      Vector3D<Precision> const &localDirection,
      const Precision stepMax);

  VECGEOM_CUDA_HEADER_BOTH
  static inline Precision ScalarDistanceToOutKernel(
      UnplacedPolyhedron const &unplaced,
      Vector3D<Precision> const &localPoint,
      Vector3D<Precision> const &localDirection,
      const Precision stepMax);

  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ScalarSafetyToInKernel(
      UnplacedPolyhedron const &unplaced,
      Vector3D<Precision> const &point,
      Precision &safety);

  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ScalarSafetyToOutKernel(
      UnplacedPolyhedron const &unplaced,
      Vector3D<Precision> const &point,
      Precision &safety);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
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
  static typename Backend::precision_v ComputeDistance(
      Quadrilaterals const &quadrilaterals,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

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

}

template <bool treatInnerT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v PolyhedronImplementation<treatInnerT>::FindZSegment(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &point) {
  return FindSegmentIndex<Backend>(
      &polyhedron.GetZPlanes()[0], polyhedron.GetZPlanes().size(), point[2]);
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v PolyhedronImplementation<treatInnerT>::FindPhiSegment(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &point) {

  typename Backend::int_v result = -1;
  typename Backend::bool_v inSection;
  SOA3D<Precision> const &phiSections = polyhedron.GetPhiSections();
  for (int i = 0, iMax = polyhedron.GetSideCount(); i < iMax; ++i) {
    inSection = point[0]*phiSections.x(i) +
                point[1]*phiSections.y(i) +
                point[2]*phiSections.z(i) >= 0 &&
                point[0]*phiSections.x(i+1) +
                point[1]*phiSections.y(i+1) +
                point[2]*phiSections.z(i+1) >= 0;
    MaskedAssign(inSection, i, &result);
    if (Backend::early_returns) {
      if (IsFull(result >= 0)) return result;
    }
  }
  return result;
}

namespace {

// Helper functions to prevent clutter in code. Calls the appropriate method
// depending on whether surface is being treated or not, casting arrays to fixed
// size for the fixed side count cases.

template <bool treatSurfaceT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename TreatSurfaceTraits<treatSurfaceT, Backend>::Surface_t
CallConvexInside(
    Quadrilaterals const &quadrilaterals,
    Vector3D<typename Backend::precision_v> const &localPoint) {
  if (treatSurfaceT) {
    return Planes::InsideKernel<Backend>(
      quadrilaterals.size(),
      quadrilaterals.GetNormals().x(),
      quadrilaterals.GetNormals().y(),
      quadrilaterals.GetNormals().z(),
      &quadrilaterals.GetDistances()[0],
      localPoint
    );
  } else {
    return Planes::ContainsKernel<Backend>(
      quadrilaterals.size(),
      quadrilaterals.GetNormals().x(),
      quadrilaterals.GetNormals().y(),
      quadrilaterals.GetNormals().z(),
      &quadrilaterals.GetDistances()[0],
      localPoint
    );
  }
}

}

template <bool treatInnerT>
template <bool treatSurfaceT>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::ScalarInsideKernel(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<Precision> const &localPoint,
    typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t &inside) {

  inside = TreatSurfaceTraits<treatSurfaceT, kScalar>::kOutside;

  // First check if in bounding tube
  {
    bool inBounds;
    HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
        UnplacedContains<kScalar>(
            polyhedron.GetBoundingTube(), localPoint, inBounds);
    if (!inBounds) return;
  }

  // // Check if inside z-bounds
  // if (localPoint[2] < polyhedron.GetTolerantZMin() ||
  //     localPoint[2] > polyhedron.GetTolerantZMax()) {
  //   return;
  // }

  // Find correct segment by checking Z-bounds
  int zIndex = FindZSegment<kScalar>(polyhedron, localPoint);

  // If an invalid segment index is returned, the point must be on the surface
  // of the endcaps (Inside) or outside (Contains)
  if (zIndex < 0 || zIndex >= polyhedron.GetZSegmentCount()) {
    if (treatSurfaceT) {
      inside = EInside::kSurface;
    }
    return;
  }

  UnplacedPolyhedron::Segment const &segment =
      polyhedron.GetZSegment(zIndex);

  inside = CallConvexInside<treatSurfaceT, kScalar>(
      segment.outer, localPoint);

  // If not in the outer shell, done
  if (inside != TreatSurfaceTraits<treatSurfaceT, kScalar>::kInside) return;

  // Otherwise check that the point is not inside the inner shell
  if (treatInnerT && segment.hasInnerRadius) {
    typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t insideInner 
        = CallConvexInside<treatSurfaceT, kScalar>(segment.inner, localPoint);
    if (treatSurfaceT) {
      if (insideInner == EInside::kSurface) {
        inside = EInside::kSurface;
      } else if (insideInner == EInside::kInside) {
        inside = EInside::kOutside;
      }
    } else {
      inside &= !insideInner;
    }
  }
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<treatInnerT>::DistanceToInZSegment(
    UnplacedPolyhedron::Segment const &segment,
    int sideCount,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;

  Float_t distance(kInfinity);

  Float_t test = Quadrilaterals::DistanceToInKernel<Backend, SOA3D<Precision> >(
      sideCount,
      segment.outer.GetNormals().x(),
      segment.outer.GetNormals().y(),
      segment.outer.GetNormals().z(),
      &segment.outer.GetDistances()[0],
      segment.outer.GetSides(),
      segment.outer.GetCorners(),
      point,
      direction);
  MaskedAssign(test >= 0, test, &distance);

  if (treatInnerT && segment.hasInnerRadius) {
    test = Quadrilaterals::DistanceToInKernel<Backend, SOA3D<Precision> >(
        sideCount,
        segment.inner.GetNormals().x(),
        segment.inner.GetNormals().y(),
        segment.inner.GetNormals().z(),
        &segment.inner.GetDistances()[0],
        segment.inner.GetSides(),
        segment.inner.GetCorners(),
        point,
        direction);
    MaskedAssign(test >= 0 && test < distance, test, &distance);
  }

  return distance;
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<treatInnerT>::DistanceToOutZSegment(
    UnplacedPolyhedron::Segment const &segment,
    int sideCount,
    Precision zMin,
    Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;

  Float_t distance(kInfinity);

  Float_t test = Quadrilaterals::DistanceToOutKernel<Backend>(
      sideCount,
      segment.outer.GetNormals().x(),
      segment.outer.GetNormals().y(),
      segment.outer.GetNormals().z(),
      &segment.outer.GetDistances()[0],
      zMin,
      zMax,
      point,
      direction);
  MaskedAssign(test >= 0, test, &distance);

  if (treatInnerT && segment.hasInnerRadius) {
    test = Quadrilaterals::DistanceToInKernel<Backend, SOA3D<Precision> >(
        sideCount,
        segment.inner.GetNormals().x(),
        segment.inner.GetNormals().y(),
        segment.inner.GetNormals().z(),
        &segment.inner.GetDistances()[0],
        segment.inner.GetSides(),
        segment.inner.GetCorners(),
        point,
        direction);
    MaskedAssign(test >= 0 && test < distance, test, &distance);
  }

  return distance;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
Precision
PolyhedronImplementation<treatInnerT>::ScalarDistanceToInKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &localPoint,
    Vector3D<Precision> const &localDirection,
    const Precision stepMax) {

  int zIndex = FindZSegment<kScalar>(unplaced, localPoint);
  // Don't go out of bounds
  const int zMax = unplaced.GetZSegmentCount();
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = localDirection[2] >= 0;

  Precision distance = kInfinity;
  if (goingRight) {
    for (int zMax = unplaced.GetZSegmentCount(); zIndex < zMax;) {
      distance = DistanceToInZSegment<kScalar>(
          unplaced.GetZSegment(zIndex),
          unplaced.GetSideCount(),
          localPoint,
          localDirection);
      if (distance >= 0 && distance < kInfinity) break;
      if (unplaced.GetZPlanes()[++zIndex] - localPoint[2] > distance) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      distance = DistanceToInZSegment<kScalar>(
          unplaced.GetZSegment(zIndex),
          unplaced.GetSideCount(),
          localPoint,
          localDirection);
      if (distance >= 0 && distance < kInfinity) break;
      if (localPoint[2] - unplaced.GetZPlanes()[zIndex] > distance) break;
    }
  }

  bool firstCap = localPoint[2] < unplaced.GetZPlanes()[0] && goingRight;
  bool secondCap =
      localPoint[2] > unplaced.GetZPlanes()[unplaced.GetZSegmentCount()] &&
      !goingRight;
  // Check endcaps
  if (firstCap || secondCap) {
    Precision distanceEndcap =
        unplaced.GetEndCaps().Distance<kScalar>(localPoint, localDirection);
    if (distanceEndcap >= 0 && distanceEndcap < distance) {
      Vector3D<Precision> intersection =
          localPoint + distanceEndcap*localDirection;
      int phiIndex = FindPhiSegment<kScalar>(unplaced, intersection);
      UnplacedPolyhedron::Segment const &segment =
          unplaced.GetZSegment(firstCap ? 0 : zMax-1);
      bool inBounds =
          segment.outer.GetNormals().x(phiIndex)*intersection[0] +
          segment.outer.GetNormals().y(phiIndex)*intersection[1] +
          segment.outer.GetNormals().z(phiIndex)*intersection[2] +
          segment.outer.GetDistances()[phiIndex] < 0;
      if (treatInnerT && inBounds && segment.hasInnerRadius) {
        inBounds &=
            segment.inner.GetNormals().x(phiIndex)*intersection[0] +
            segment.inner.GetNormals().y(phiIndex)*intersection[1] +
            segment.inner.GetNormals().z(phiIndex)*intersection[2] +
            segment.inner.GetDistances()[phiIndex] >= 0;
      }
      distance = inBounds ? distanceEndcap : distance;
    }
  }

  return distance < stepMax ? distance : stepMax;
}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::ScalarSafetyToInKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point,
    Precision &safety) {

  // Safety in Z-axis

  safety = unplaced.GetZPlanes()[0] - point[2];
  if (safety > 0) {
    safety = Max<Precision>(
                 Max<Precision>(safety,
                     Abs(point[0] - unplaced.GetStartCapOuterRadius())),
                 Abs(point[1] - unplaced.GetStartCapOuterRadius()));
    return;
  }

  safety = point[2] - unplaced.GetZPlanes()[unplaced.GetZSegmentCount()];
  if (safety > 0) {
    safety = Max<Precision>(
                 Max<Precision>(safety,
                     Abs(point[0] - unplaced.GetEndCapOuterRadius())),
                 Abs(point[1] - unplaced.GetEndCapOuterRadius()));
    return;
  }

  // Safety in rho/phi

  int zIndex = FindZSegment<kScalar>(unplaced, point);
  int phiIndex = FindPhiSegment<kScalar>(unplaced, point);
  // Now it is known exactly where to look

  UnplacedPolyhedron::Segment const &segment = unplaced.GetZSegment(zIndex);
  safety = segment.outer.GetNormals().x(phiIndex)*point[0] +
           segment.outer.GetNormals().y(phiIndex)*point[1] +
           segment.outer.GetNormals().z(phiIndex)*point[2] +
           segment.outer.GetDistances()[phiIndex];
  if (treatInnerT && safety < 0 && segment.hasInnerRadius) {
    safety = -(segment.inner.GetNormals().x(phiIndex)*point[0] +
               segment.inner.GetNormals().y(phiIndex)*point[1] +
               segment.inner.GetNormals().z(phiIndex)*point[2] +
               segment.inner.GetDistances()[phiIndex]);
  }

}

template <bool treatInnerT>
VECGEOM_CUDA_HEADER_BOTH
Precision
PolyhedronImplementation<treatInnerT>::ScalarDistanceToOutKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) {

  int zIndex = FindZSegment<kScalar>(unplaced, point);
  // Don't go out of bounds
  const int zMax = unplaced.GetZSegmentCount();
  zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax-1 : zIndex);

  // Traverse Z-segments left or right depending on sign of direction
  bool goingRight = direction[2] >= 0;

  Precision distance =
      unplaced.GetEndCaps().Distance<kScalar>(point, direction);
  if (goingRight) {
    for (int zMax = unplaced.GetZSegmentCount(); zIndex < zMax;) {
      Precision test = DistanceToOutZSegment<kScalar>(
          unplaced.GetZSegment(zIndex),
          unplaced.GetSideCount(),
          unplaced.GetZPlanes()[zIndex],
          unplaced.GetZPlanes()[zMax],
          point,
          direction);
      if (test < distance) {
        distance = test;
        break;
      }
      if (unplaced.GetZPlanes()[++zIndex] - point[2] > distance) break;
    }
  } else {
    // Going left
    for (; zIndex >= 0; --zIndex) {
      Precision test = DistanceToOutZSegment<kScalar>(
          unplaced.GetZSegment(zIndex),
          unplaced.GetSideCount(),
          unplaced.GetZPlanes()[zIndex],
          unplaced.GetZPlanes()[zMax],
          point,
          direction);
      if (test < distance) {
        distance = test;
        break;
      }
      if (point[2] - unplaced.GetZPlanes()[zIndex] > distance) break;
    }
  }

  return distance < stepMax ? distance : stepMax;
}

template <bool treatInnerT>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::ScalarSafetyToOutKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point,
    Precision &safety) {

  // Safety in Z-axis

  Precision testSafety;

  testSafety = point[2] - unplaced.GetZPlanes()[0];
  if (testSafety >= 0) safety = testSafety;

  testSafety = unplaced.GetZPlanes()[unplaced.GetZSegmentCount()] - point[2];
  if (testSafety >= 0) safety = Min<Precision>(safety, testSafety);

  // Safety in rho/phi

  int zIndex = FindZSegment<kScalar>(unplaced, point);
  int phiIndex = FindPhiSegment<kScalar>(unplaced, point);
  // Now it is known exactly where to look

  UnplacedPolyhedron::Segment const &segment = unplaced.GetZSegment(zIndex);
  testSafety = -(segment.outer.GetNormals().x(phiIndex)*point[0] +
                 segment.outer.GetNormals().y(phiIndex)*point[1] +
                 segment.outer.GetNormals().z(phiIndex)*point[2] +
                 segment.outer.GetDistances()[phiIndex]);
  if (treatInnerT && testSafety < 0 && segment.hasInnerRadius) {
    testSafety = segment.inner.GetNormals().x(phiIndex)*point[0] +
             segment.inner.GetNormals().y(phiIndex)*point[1] +
             segment.inner.GetNormals().z(phiIndex)*point[2] +
             segment.inner.GetDistances()[phiIndex];
  }

  if (testSafety >= 0) safety = Min<Precision>(safety, testSafety);
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::UnplacedContains(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Not implemented.\n");
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
  Assert(0, "Contains not implemented.\n");
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "Inside not implemented.\n");
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

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  distance = kInfinity;

  Vector3D<Float_t> localPoint = transformation.Transform(point);
  Vector3D<Float_t> localDirection = transformation.Transform(direction);

  // if (Backend::early_returns) {
  //   Float_t hitsTube;
  //   HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
  //       DistanceToIn<Backend>(
  //           unplaced.GetBoundingTube(), Transformation3D::kIdentity,
  //           localPoint, localDirection, stepMax, hitsTube);
  //   if (IsFull(hitsTube == kInfinity)) return;
  // }

  Float_t distanceResult;
  for (int i = 0, iMax = unplaced.GetZSegmentCount(); i < iMax; ++i) {
    UnplacedPolyhedron::Segment const &s = unplaced.GetZSegment(i);
    distanceResult =
        Quadrilaterals::DistanceToInKernel<Backend, SOA3D<Precision> >(
            unplaced.GetSideCount(),
            s.outer.GetNormals().x(),
            s.outer.GetNormals().y(),
            s.outer.GetNormals().z(),
            &s.outer.GetDistances()[0],
            s.outer.GetSides(),
            s.outer.GetCorners(),
            localPoint,
            localDirection);
    MaskedAssign(distanceResult < distance, distanceResult, &distance);
    if (treatInnerT) {
      if (s.hasInnerRadius) {
        distanceResult =
            Quadrilaterals::DistanceToInKernel<Backend, SOA3D<Precision> >(
                unplaced.GetSideCount(),
                s.inner.GetNormals().x(),
                s.inner.GetNormals().y(),
                s.inner.GetNormals().z(),
                &s.inner.GetDistances()[0],
                s.inner.GetSides(),
                s.inner.GetCorners(),
                localPoint,
                localDirection);
        MaskedAssign(distanceResult < distance, distanceResult, &distance);
      }
    }
    // If the next segment is further away than the best distance, no better
    // distance can be found.
    if (Backend::early_returns) {
      if (IsFull(Abs(localPoint[2] - unplaced.GetZPlane(i+1)) > distance)) {
        break;
      }
    }
  }

  // Distance to endcaps
  distanceResult =
      unplaced.GetEndCaps().Distance<Backend>(localPoint, localDirection);
  Bool_t endcapValid = distanceResult < distance;
  if (Any(endcapValid)) {
    Vector3D<Float_t> intersection = localPoint + distanceResult*localDirection;
    Bool_t thisCap = endcapValid && localPoint[2] < unplaced.GetZPlanes()[0];
    UnplacedPolyhedron::Segment const *s = &unplaced.GetZSegment(0);
    if (Any(thisCap)) {
      Bool_t insideCap =
          Planes::ContainsKernel<Backend>(
              unplaced.GetSideCount(),
              s->outer.GetNormals().x(),
              s->outer.GetNormals().y(),
              s->outer.GetNormals().z(),
              &s->outer.GetDistances()[0],
              intersection);
      if (treatInnerT && Any(insideCap) && s->hasInnerRadius) {
        insideCap &=
            !(Planes::ContainsKernel<Backend>(
                unplaced.GetSideCount(),
                s->inner.GetNormals().x(),
                s->inner.GetNormals().y(),
                s->inner.GetNormals().z(),
                &s->inner.GetDistances()[0],
                intersection));
      }
      MaskedAssign(thisCap && insideCap, distanceResult, &distance);
    }
    s = &unplaced.GetZSegment(unplaced.GetZSegmentCount()-1);
    thisCap = endcapValid && localPoint[2] >
              unplaced.GetZPlanes()[unplaced.GetZSegmentCount()];
    if (Any(thisCap)) {
      Bool_t insideCap =
          Planes::ContainsKernel<Backend>(
              unplaced.GetSideCount(),
              s->outer.GetNormals().x(),
              s->outer.GetNormals().y(),
              s->outer.GetNormals().z(),
              &s->outer.GetDistances()[0],
              intersection);
      if (treatInnerT && Any(insideCap) && s->hasInnerRadius) {
        insideCap &=
            !(Planes::ContainsKernel<Backend>(
                unplaced.GetSideCount(),
                s->inner.GetNormals().x(),
                s->inner.GetNormals().y(),
                s->inner.GetNormals().z(),
                &s->inner.GetDistances()[0],
                intersection));
      }
      MaskedAssign(thisCap && insideCap, distanceResult, &distance);
    }
  }

  // Impose upper limit of stepMax on the result
  MaskedAssign(distance > stepMax && distance < kInfinity, stepMax, &distance);
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
//   Assert(0, "Not implemented.\n");
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

  typedef typename Backend::precision_v Float_t;

  distance = kInfinity;

  Float_t distanceResult;
  for (int i = 0, iMax = unplaced.GetZSegmentCount(); i < iMax; ++i) {
    UnplacedPolyhedron::Segment const &s = unplaced.GetZSegment(i);
    distanceResult = Quadrilaterals::DistanceToOutKernel<Backend>(
        unplaced.GetSideCount(),
        s.outer.GetNormals().x(),
        s.outer.GetNormals().y(),
        s.outer.GetNormals().z(),
        &s.outer.GetDistances()[0],
        unplaced.GetZPlanes()[i],
        unplaced.GetZPlanes()[i+1],
        point,
        direction);
    MaskedAssign(distanceResult < distance, distanceResult, &distance);
    if (treatInnerT) {
      if (s.hasInnerRadius) {
        distanceResult =
            Quadrilaterals::DistanceToInKernel<Backend, SOA3D<Precision> > (
                unplaced.GetSideCount(),
                s.inner.GetNormals().x(),
                s.inner.GetNormals().y(),
                s.inner.GetNormals().z(),
                &s.inner.GetDistances()[0],
                s.inner.GetSides(),
                s.inner.GetCorners(),
                point,
                direction);
        MaskedAssign(distanceResult < distance, distanceResult, &distance);
      }
    }
    // If the next segment is further away than the best distance, no better
    // distance can be found.
    if (Backend::early_returns) {
      if (IsFull(Abs(point[2] - unplaced.GetZPlane(i+1)) > distance)) break;
    }
  }
  distanceResult =
      unplaced.GetEndCaps().Distance<Backend>(point, direction);
  MaskedAssign(distanceResult < distance, distanceResult, &distance);

  MaskedAssign(distance > stepMax && distance < kInfinity, stepMax, &distance);
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "Not implemented.\n");
}

template <bool treatInnerT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "Not implemented.\n");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
