/// \file PolyhedronImplementation.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedPolyhedron.h"

namespace VECGEOM_NAMESPACE {

template <class PolyhedronType>
struct PolyhedronImplementation {

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void FindPhiSegment(
      typename Backend::precision_v phi0,
      UnplacedPolyhedron const &polyhedron,
      typename Backend::int_v side);

  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void InsideSegments(
      UnplacedPolyhedron const &unplaced,
      size_t segmentIndex,
      Transformation3D const &transformation,
      Vector3D<Precision> const &point,
      typename Backend::inside_v &inside,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToSegments(
      UnplacedPolyhedron const &unplaced,
      size_t segmentIndex,
      Vector3D<Precision> const &point,
      typename Backend::precision_v &normal,
      typename Backend::precision_v &distance);

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

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      Vector3D<Precision> const &polyhedronDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      Vector3D<Precision> const &polyhedronDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

}; // End struct PolyhedronImplementation

template <class PolyhedronType>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::FindPhiSegment(
    typename Backend::precision_v phi0,
    UnplacedPolyhedron const &polyhedron,
    typename Backend::int_v side) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t phi = GenericKernels<Backend>::NormalizeAngle(
      phi0 - polyhedron.GetPhiStart());
  side = Int_t(phi / polyhedron.GetPhiDelta());
  if (PolyhedronType::phiTreatment) {
    Bool_t inPhi = side > polyhedron.GetSideCount();
    if (inPhi == Backend::kZero) return;
    phi = GenericKernels<Backend>::NormalizeAngle(phi);
    Float_t start = polyhedron.GetPhiStart() - phi;
    Float_t end = phi - polyhedron.GetPhiEnd();
    MaskedAssign(inPhi && start < end, 0, side);
    MaskedAssign(inPhi && start >= end, polyhedron.GetSideCount()-1);
  }
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::InsideSegments(
    UnplacedPolyhedron const &unplaced,
    size_t segmentIndex,
    Transformation3D const &transformation,
    Vector3D<Precision> const &point,
    typename Backend::inside_v &inside,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;

  Vector3D<Precision> localPoint = transformation.Transform(point);

  int segment = FindPhiSegment<kScalar>(point.Phi());

  Float_t normal;
  DistanceToSegments(unplaced, segmentIndex, point, distance, normal);

  inside = EInside::kOutside;
  MaskedAssign(normal < 0., EInside::kInside, inside);
  MaskedAssign((Abs(normal) < kTolerance) && (distance < 2.*kTolerance),
               EInside::kSurface, inside);
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::DistanceToSegments(
    UnplacedPolyhedron const &unplaced,
    size_t segmentIndex,
    Vector3D<Precision> const &point,
    typename Backend::precision_v &normal,
    typename Backend::precision_v &distance) {
  // NYI
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_