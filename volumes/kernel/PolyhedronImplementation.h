/// \file PolyhedronImplementation.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/Quadrilaterals.h"
#include "volumes/UnplacedPolyhedron.h"

namespace VECGEOM_NAMESPACE {

template <bool treatInnerT, unsigned sideCountT>
struct PolyhedronImplementation {

  template <bool treatSurfaceT>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ScalarInsideKernel(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<Precision> const &localPoint,
      typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t &inside);

  template<class Backend>
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

template <unsigned sideCountT, bool treatSurfaceT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename TreatSurfaceTraits<treatSurfaceT, Backend>::Surface_t
CallConvexInside(
    Quadrilaterals<0> const &quadrilaterals,
    Vector3D<typename Backend::precision_v> const &localPoint) {
  if (treatSurfaceT) {
    return Planes<sideCountT>::template InsideKernel<kScalar>(
      quadrilaterals.size(),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().x()),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().y()),
      Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().z()),
      Quadrilaterals<sideCountT>::ToFixedSize(&quadrilaterals.GetDistance()[0]),
      localPoint
    );
  } else {
    return Planes<sideCountT>::template ContainsKernel<kScalar>(
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

  // Check if inside z-bounds (tolerance is included in Z-min/max)
  if (localPoint[2] < polyhedron.GetZMin() ||
      localPoint[2] > polyhedron.GetZMax()) {
    return;
  }

  Array<UnplacedPolyhedron::Segment> const &segments = polyhedron.GetSegments();
  Array<UnplacedPolyhedron::Segment>::const_iterator s = segments.cbegin();
  Array<UnplacedPolyhedron::Segment>::const_iterator sEnd = segments.cend();
  // Find correct segment by checking z-bounds
  while (s != sEnd && localPoint[2] > s->zMax) ++s;
  if (s == sEnd) return;

  Quadrilaterals<0> const &outer = s->outer;

  inside = CallConvexInside<sideCountT, treatSurfaceT, kScalar>(
      outer, localPoint);

  // If not in the outer shell, done
  if (inside != TreatSurfaceTraits<treatSurfaceT, kScalar>::kInside) return;

  if (treatInnerT) {
    Quadrilaterals<0> const &inner = s->inner;
    typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t insideInner 
        = CallConvexInside<sideCountT, treatSurfaceT, kScalar>(
            inner, localPoint);
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
  Assert(0, "DistanceToIn not implemented.\n");
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<treatInnerT, sideCountT>::ComputeDistance(
    Quadrilaterals<0> const &quadrilaterals,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {
  if (sideCountT > 0) {
    return Planes<sideCountT>::template DistanceToOutKernel<Backend>(
        sideCountT,
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().x()),
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().y()),
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().z()),
        Quadrilaterals<sideCountT>::ToFixedSize(
            &quadrilaterals.GetDistance()[0]),
        point,
        direction);
  } else {
    return Planes<0>::template DistanceToOutKernel<Backend>(
        quadrilaterals.size(),
        quadrilaterals.GetNormal().x(),
        quadrilaterals.GetNormal().y(),
        quadrilaterals.GetNormal().z(),
        &quadrilaterals.GetDistance()[0],
        point,
        direction);
  }
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

  Float_t distanceResult;
  distance = kInfinity;
  Array<UnplacedPolyhedron::Segment> const &segments = unplaced.GetSegments();

  for (Array<UnplacedPolyhedron::Segment>::const_iterator s = segments.cbegin(),
       sEnd = segments.cend(); s != sEnd; ++s) {
    distanceResult = ComputeDistance<Backend>(s->outer, point, direction);
    MaskedAssign(distanceResult < distance && distanceResult <= stepMax,
                 distanceResult, &distance);
    if (treatInnerT) {
      if (s->hasInnerRadius) {
        distanceResult = ComputeDistance<Backend>(s->inner, point, direction);
        MaskedAssign(distanceResult < distance && distanceResult <= stepMax,
                     distanceResult, &distance);
      }
    }
  }
  distanceResult =
      unplaced.GetEndCaps().DistanceToOut<Backend>(point, direction);
  MaskedAssign(distanceResult < distance && distanceResult <= stepMax,
               distanceResult, &distance);
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "SafetyToIn not implemented.\n");
}

template <bool treatInnerT, unsigned sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<treatInnerT, sideCountT>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "SafetyToOut not implemented.\n");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_