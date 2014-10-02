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

template <bool treatInnerT, unsigned sideCountT>
struct PolyhedronImplementation {

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::int_v FindZSegment(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::int_v FindPhiSegment(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &point);

  template <bool treatSurfaceT>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ScalarInsideKernel(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &localPoint,
      typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t &inside);

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
      Quadrilaterals<0> const &quadrilaterals,
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

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v
PolyhedronImplementation<treatInnerT, sideCountT>::FindZSegment(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &point) {
  return FindSegmentIndex<Backend>(
      &polyhedron.GetZPlanes()[0], polyhedron.GetZPlanes().size(), point[2]);
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v
PolyhedronImplementation<treatInnerT, sideCountT>::FindPhiSegment(
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

template <unsigned sideCountT, bool treatSurfaceT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename TreatSurfaceTraits<treatSurfaceT, Backend>::Surface_t
CallConvexInside(
    Quadrilaterals<0> const &quadrilaterals,
    Vector3D<typename Backend::precision_v> const &localPoint) {
  if (treatSurfaceT) {
    return Planes<sideCountT>::template InsideKernel<Backend>(
      quadrilaterals.size(),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().x()),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().y()),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().z()),
      Quadrilaterals<sideCountT>::ToFixedSize(&quadrilaterals.GetDistance()[0]),
      localPoint
    );
  } else {
    return Planes<sideCountT>::template ContainsKernel<Backend>(
      quadrilaterals.size(),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().x()),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().y()),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().z()),
      Quadrilaterals<sideCountT>::ToFixedSize(&quadrilaterals.GetDistance()[0]),
      localPoint
    );
  }
}

}

template <bool treatInnerT, unsigned sideCountT>
template <bool treatSurfaceT>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::ScalarInsideKernel(
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
  if (zIndex < 0 || zIndex >= polyhedron.GetSegmentCount()) {
    if (treatSurfaceT) {
      inside = EInside::kSurface;
    }
    return;
  }

  UnplacedPolyhedron::Segment const &segment =
      polyhedron.GetSegment(zIndex);

  inside =
      CallConvexInside<sideCountT, treatSurfaceT, kScalar>(
          segment.outer, localPoint);

  // If not in the outer shell, done
  if (inside != TreatSurfaceTraits<treatSurfaceT, kScalar>::kInside) return;

  // Otherwise check that the point is not inside the inner shell
  if (treatInnerT && segment.hasInnerRadius) {
    typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t insideInner 
        = CallConvexInside<sideCountT, treatSurfaceT, kScalar>(
            segment.inner, localPoint);
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

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToInKernel(
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

  safety = point[2] - unplaced.GetZPlanes()[unplaced.GetSegmentCount()];
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

  UnplacedPolyhedron::Segment const &segment = unplaced.GetSegment(zIndex);
  safety = segment.outer.GetNormal().x(phiIndex)*point[0] +
           segment.outer.GetNormal().y(phiIndex)*point[1] +
           segment.outer.GetNormal().z(phiIndex)*point[2] +
           segment.outer.GetDistance()[phiIndex];
  if (treatInnerT && safety < 0 && segment.hasInnerRadius) {
    safety = -(segment.inner.GetNormal().x(phiIndex)*point[0] +
               segment.inner.GetNormal().y(phiIndex)*point[1] +
               segment.inner.GetNormal().z(phiIndex)*point[2] +
               segment.inner.GetDistance()[phiIndex]);
  }

}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToOutKernel(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &point,
    Precision &safety) {

  // Safety in Z-axis

  Precision testSafety;

  testSafety = point[2] - unplaced.GetZPlanes()[0];
  if (testSafety >= 0) safety = testSafety;

  testSafety = unplaced.GetZPlanes()[unplaced.GetSegmentCount()] - point[2];
  if (testSafety >= 0) safety = Min<Precision>(safety, testSafety);

  // Safety in rho/phi

  int zIndex = FindZSegment<kScalar>(unplaced, point);
  int phiIndex = FindPhiSegment<kScalar>(unplaced, point);
  // Now it is known exactly where to look

  UnplacedPolyhedron::Segment const &segment = unplaced.GetSegment(zIndex);
  testSafety = -(segment.outer.GetNormal().x(phiIndex)*point[0] +
                 segment.outer.GetNormal().y(phiIndex)*point[1] +
                 segment.outer.GetNormal().z(phiIndex)*point[2] +
                 segment.outer.GetDistance()[phiIndex]);
  if (treatInnerT && testSafety < 0 && segment.hasInnerRadius) {
    testSafety = segment.inner.GetNormal().x(phiIndex)*point[0] +
             segment.inner.GetNormal().y(phiIndex)*point[1] +
             segment.inner.GetNormal().z(phiIndex)*point[2] +
             segment.inner.GetDistance()[phiIndex];
  }

  if (testSafety >= 0) safety = Min<Precision>(safety, testSafety);
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::UnplacedContains(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Not implemented.\n");
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::Contains(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Contains not implemented.\n");
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "Inside not implemented.\n");
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::DistanceToIn(
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

  if (Backend::early_returns) {
    Float_t hitsTube;
    HasInnerRadiiTraits<treatInnerT>::TubeKernels::template
        DistanceToIn<Backend>(
            unplaced.GetBoundingTube(), Transformation3D::kIdentity,
            localPoint, localDirection, stepMax, hitsTube);
    if (IsFull(hitsTube == kInfinity)) return;
  }

  Float_t distanceResult;
  for (int i = 0, iMax = unplaced.GetSegmentCount(); i < iMax; ++i) {
    UnplacedPolyhedron::Segment const &s = unplaced.GetSegment(i);
    distanceResult = Quadrilaterals<sideCountT>::template
        DistanceToInKernel<Backend, SOA3D<Precision> >(
            unplaced.GetSideCount(),
            Quadrilaterals<sideCountT>::ToFixedSize(
                s.outer.GetNormal().x()),
            Quadrilaterals<sideCountT>::ToFixedSize(
                s.outer.GetNormal().y()),
            Quadrilaterals<sideCountT>::ToFixedSize(
                s.outer.GetNormal().z()),
            Quadrilaterals<sideCountT>::ToFixedSize(
                &s.outer.GetDistance()[0]),
            s.outer.GetSides(),
            s.outer.GetCorners(),
            localPoint,
            localDirection);
    MaskedAssign(distanceResult < distance, distanceResult, &distance);
    if (treatInnerT) {
      if (s.hasInnerRadius) {
        distanceResult = Quadrilaterals<sideCountT>::template
            DistanceToInKernel<Backend, SOA3D<Precision> >(
                unplaced.GetSideCount(),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s.inner.GetNormal().x()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s.inner.GetNormal().y()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s.inner.GetNormal().z()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    &s.inner.GetDistance()[0]),
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
      if (IsFull(Abs(localPoint[2] - s.zMax) > distance)) break;
    }
  }

  // Distance to endcaps
  distanceResult =
      unplaced.GetEndCaps().DistanceToIn<Backend>(localPoint, localDirection);
  Bool_t endcapValid = distanceResult < distance;
  if (Any(endcapValid)) {
    Vector3D<Float_t> intersection = localPoint + distanceResult*localDirection;
    Bool_t thisCap = endcapValid && localPoint[2] < unplaced.GetZPlanes()[0];
    UnplacedPolyhedron::Segment const *s = &unplaced.GetSegment(0);
    if (Any(thisCap)) {
      Bool_t insideCap =
          Planes<sideCountT>::template ContainsKernel<Backend>(
              unplaced.GetSideCount(),
              Quadrilaterals<sideCountT>::ToFixedSize(s->outer.GetNormal().x()),
              Quadrilaterals<sideCountT>::ToFixedSize(s->outer.GetNormal().y()),
              Quadrilaterals<sideCountT>::ToFixedSize(s->outer.GetNormal().z()),
              Quadrilaterals<sideCountT>::ToFixedSize(
                  &s->outer.GetDistance()[0]),
              intersection);
      if (treatInnerT && Any(insideCap) && s->hasInnerRadius) {
        insideCap &=
            !(Planes<sideCountT>::template ContainsKernel<Backend>(
                unplaced.GetSideCount(),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s->inner.GetNormal().x()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s->inner.GetNormal().y()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s->inner.GetNormal().z()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    &s->inner.GetDistance()[0]),
                intersection));
      }
      MaskedAssign(thisCap && insideCap, distanceResult, &distance);
    }
    s = &unplaced.GetSegment(unplaced.GetSegmentCount()-1);
    thisCap = endcapValid && localPoint[2] >
              unplaced.GetZPlanes()[unplaced.GetSegmentCount()];
    if (Any(thisCap)) {
      Bool_t insideCap =
          Planes<sideCountT>::template ContainsKernel<Backend>(
              unplaced.GetSideCount(),
              Quadrilaterals<sideCountT>::ToFixedSize(s->outer.GetNormal().x()),
              Quadrilaterals<sideCountT>::ToFixedSize(s->outer.GetNormal().y()),
              Quadrilaterals<sideCountT>::ToFixedSize(s->outer.GetNormal().z()),
              Quadrilaterals<sideCountT>::ToFixedSize(
                  &s->outer.GetDistance()[0]),
              intersection);
      if (treatInnerT && Any(insideCap) && s->hasInnerRadius) {
        insideCap &=
            !(Planes<sideCountT>::template ContainsKernel<Backend>(
                unplaced.GetSideCount(),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s->inner.GetNormal().x()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s->inner.GetNormal().y()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s->inner.GetNormal().z()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    &s->inner.GetDistance()[0]),
                intersection));
      }
      MaskedAssign(thisCap && insideCap, distanceResult, &distance);
    }
  }

  // Impose upper limit of stepMax on the result
  MaskedAssign(distance > stepMax && distance < kInfinity, stepMax, &distance);
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::DistanceToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;

  distance = kInfinity;

  Float_t distanceResult;
  for (int i = 0, iMax = unplaced.GetSegmentCount(); i < iMax; ++i) {
    UnplacedPolyhedron::Segment const &s = unplaced.GetSegment(i);
    distanceResult = Quadrilaterals<sideCountT>::template
        DistanceToOutKernel<Backend>(
            unplaced.GetSideCount(),
            Quadrilaterals<sideCountT>::ToFixedSize(
                s.outer.GetNormal().x()),
            Quadrilaterals<sideCountT>::ToFixedSize(
                s.outer.GetNormal().y()),
            Quadrilaterals<sideCountT>::ToFixedSize(
                s.outer.GetNormal().z()),
            Quadrilaterals<sideCountT>::ToFixedSize(
                &s.outer.GetDistance()[0]),
            unplaced.GetZPlanes()[i],
            unplaced.GetZPlanes()[i+1],
            point,
            direction);
    MaskedAssign(distanceResult < distance, distanceResult, &distance);
    if (treatInnerT) {
      if (s.hasInnerRadius) {
        distanceResult = Quadrilaterals<sideCountT>::template
            DistanceToInKernel<Backend, SOA3D<Precision> > (
                unplaced.GetSideCount(),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s.inner.GetNormal().x()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s.inner.GetNormal().y()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    s.inner.GetNormal().z()),
                Quadrilaterals<sideCountT>::ToFixedSize(
                    &s.inner.GetDistance()[0]),
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
      if (IsFull(Abs(point[2] - s.zMax) > distance)) break;
    }
  }
  distanceResult =
      unplaced.GetEndCaps().DistanceToOut<Backend>(point, direction);
  MaskedAssign(distanceResult < distance, distanceResult, &distance);

  MaskedAssign(distance > stepMax && distance < kInfinity, stepMax, &distance);
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "Not implemented.\n");
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT, sideCountT>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "Not implemented.\n");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
