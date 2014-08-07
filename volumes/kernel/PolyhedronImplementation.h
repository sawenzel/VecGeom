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

template <bool treatInnerT, int sideCountT>
struct PolyhedronImplementation {

  template<class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
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

template <bool treatInnerT, int sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::UnplacedContains(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Not implemented.\n");
}

template <bool treatInnerT, int sideCountT>
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

template <bool treatInnerT, int sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "Inside not implemented.\n");
}

template <bool treatInnerT, int sideCountT>
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

template <bool treatInnerT, int sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v
PolyhedronImplementation<treatInnerT, sideCountT>::ComputeDistance(
    Quadrilaterals<0> const &quadrilaterals,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  if (sideCountT != 0) {
    return Planes<sideCountT>::template DistanceToOutKernel<Backend>(
        sideCountT,
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().x()),
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().y()),
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetNormal().z()),
        Quadrilaterals<sideCountT>::ToFixedSize(quadrilaterals.GetDistance()),
        point,
        direction);
  } else {
    return Planes<0>::template DistanceToOutKernel<Backend>(
        quadrilaterals.size(),
        quadrilaterals.GetNormal().x(),
        quadrilaterals.GetNormal().y(),
        quadrilaterals.GetNormal().z(),
        quadrilaterals.GetDistance(),
        point,
        direction);
  }
}

template <bool treatInnerT, int sideCountT>
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
    Quadrilaterals<0> const &outer = s->outer;
    distanceResult = ComputeDistance<Backend>(outer, point, direction);
    MaskedAssign(distanceResult < distance && distanceResult <= stepMax,
                 distanceResult, &distance);
    if (treatInnerT) {
      if (s->hasInnerRadius) {
        Quadrilaterals<0> const &inner = s->inner;
        distanceResult = ComputeDistance<Backend>(inner, point, direction);
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

template <bool treatInnerT, int sideCountT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<treatInnerT, sideCountT>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "SafetyToIn not implemented.\n");
}

template <bool treatInnerT, int sideCountT>
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