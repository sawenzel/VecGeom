/// @file BoxImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_

#include "backend/backend.h"
#include "base/vector3d.h"
#include "volumes/UnplacedBox.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BoxImplementation {

  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedBox const &box,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedBox const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedBox const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedBox const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedBox const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedBox const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedBox const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      Vector3D<Precision> const &boxDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      Vector3D<Precision> const &boxDimensions,
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

}; // End struct BoxImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBox const &box,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>(box.dimensions(), localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  InsideKernel<Backend>(unplaced.dimensions(),
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToInKernel<Backend>(
    unplaced.dimensions(),
    transformation.Transform<transCodeT, rotCodeT>(point),
    transformation.TransformDirection<rotCodeT>(direction),
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedBox const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Backend>(
    unplaced.dimensions(),
    point,
    direction,
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BoxImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToInKernel<Backend>(
    unplaced.dimensions(),
    transformation.Transform<transCodeT, rotCodeT>(point),
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BoxImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedBox const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToOutKernel<Backend>(
    unplaced.dimensions(),
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::ContainsKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  Vector3D<typename Backend::bool_v> insideDim = Backend::kFalse;
  for (int i = 0; i < 3; ++i) {
    insideDim[i] = Abs(localPoint[i]) < dimensions[i];
    if (Backend::early_returns) {
      if (!insideDim[i]) {
        inside = Backend::kFalse;
        return;
      }
    }
  }
  if (Backend::early_returns) {
    inside = Backend::kTrue;
  } else {
    inside = insideDim[0] && insideDim[1] && insideDim[2];
  }

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::InsideKernel(
    Vector3D<Precision> const &boxDimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v      Bool_t;

  Bool_t done(false);
  inside = EInside::kOutside;
  Vector3D<Float_t> pointAbs = point.Abs();

  // Check if points are outside all surfaces
  Vector3D<Bool_t> insideOuterTolerance = pointAbs <= boxDimensions
                                                      + kHalfTolerance;
  Bool_t isOutside = !insideOuterTolerance[0] &&
                     !insideOuterTolerance[1] &&
                     !insideOuterTolerance[2];
  done |= isOutside;
  if (done == Backend::kTrue) return;

  // Check if points are inside all surfaces
  Vector3D<Bool_t> insideInnerTolerance = pointAbs <= boxDimensions
                                                      - kHalfTolerance;
  Bool_t isInside = insideInnerTolerance[0] &&
                    insideInnerTolerance[1] &&
                    insideInnerTolerance[2];
  MaskedAssign(isInside, EInside::kInside, &inside);
  done |= isInside;
  if (done == Backend::kTrue) return;

  // Check for remaining surface cases
  Vector3D<Bool_t> onSurface = insideOuterTolerance && !insideInnerTolerance;
  Bool_t isSurface = (onSurface[0] && insideOuterTolerance[1]
                      && insideOuterTolerance[2]) ||
                     (onSurface[1] && insideOuterTolerance[0]
                      && insideOuterTolerance[2]) ||
                     (onSurface[2] && insideOuterTolerance[0]
                      && insideOuterTolerance[1]);
  MaskedAssign(isSurface, EInside::kSurface, &inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v      Bool_t;

  Vector3D<Float_t> safety;
  Bool_t done = Backend::kFalse;

  safety[0] = Abs(point[0]) - dimensions[0];
  safety[1] = Abs(point[1]) - dimensions[1];
  safety[2] = Abs(point[2]) - dimensions[2];

  done |= (safety[0] >= stepMax ||
           safety[1] >= stepMax ||
           safety[2] >= stepMax);
  if (done == Backend::kTrue) return;

  Float_t next, coord1, coord2;
  Bool_t hit;

  // x
  next = safety[0] / Abs(direction[0] + kTiny);
  coord1 = point[1] + next * direction[1];
  coord2 = point[2] + next * direction[2];
  hit = safety[0] >= 0 &&
        point[0] * direction[0] < 0 &&
        Abs(coord1) <= dimensions[1] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, &distance);
  done |= hit;
  if (done == Backend::kTrue) return;

  // y
  next = safety[1] / Abs(direction[1] + kTiny);
  coord1 = point[0] + next * direction[0];
  coord2 = point[2] + next * direction[2];
  hit = safety[1] >= 0 &&
        point[1] * direction[1] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, &distance);
  done |= hit;
  if (done == Backend::kTrue) return;

  // z
  next = safety[2] / Abs(direction[2] + kTiny);
  coord1 = point[0] + next * direction[0];
  coord2 = point[1] + next * direction[1];
  hit = safety[2] >= 0 &&
        point[2] * direction[2] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[1];
  MaskedAssign(!done && hit, next, &distance);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Vector3D<Float_t> safety;
    Bool_t inside;

    distance = kInfinity;

    safety[0] = Abs(point[0]) - dimensions[0];
    safety[1] = Abs(point[1]) - dimensions[1];
    safety[2] = Abs(point[2]) - dimensions[2];

    inside = safety[0] < stepMax &&
             safety[1] < stepMax &&
             safety[2] < stepMax;
    if (inside == Backend::kFalse) return;

    Vector3D<Float_t> inverseDirection = Vector3D<Float_t>(
      1. / (direction[0] + kTiny),
      1. / (direction[1] + kTiny),
      1. / (direction[2] + kTiny)
    );
    Vector3D<Float_t> distances = Vector3D<Float_t>(
      (dimensions[0] - point[0]) * inverseDirection[0],
      (dimensions[1] - point[1]) * inverseDirection[1],
      (dimensions[2] - point[2]) * inverseDirection[2]
    );

    MaskedAssign(direction[0] < 0,
                 (-dimensions[0] - point[0]) * inverseDirection[0],
                 &distances[0]);
    MaskedAssign(direction[1] < 0,
                 (-dimensions[1] - point[1]) * inverseDirection[1],
                 &distances[1]);
    MaskedAssign(direction[2] < 0,
                 (-dimensions[2] - point[2]) * inverseDirection[2],
                 &distances[2]);

    distance = distances[0];
    MaskedAssign(distances[1] < distance, distances[1], &distance);
    MaskedAssign(distances[2] < distance, distances[2], &distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  safety = -dimensions[0] + Abs(point[0]);
  Float_t safetyY = -dimensions[1] + Abs(point[1]);
  Float_t safetyZ = -dimensions[2] + Abs(point[2]);

  // TODO: check if we should use MIN/MAX here instead
  MaskedAssign(safetyY > safety, safetyY, &safety);
  MaskedAssign(safetyZ > safety, safetyZ, &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::precision_v Float_t;

   safety = dimensions[0] - Abs(point[0]);
   Float_t safetyY = dimensions[1] - Abs(point[1]);
   Float_t safetyZ = dimensions[2] - Abs(point[2]);

   // TODO: check if we should use MIN here instead
   MaskedAssign(safetyY < safety, safetyY, &safety);
   MaskedAssign(safetyZ < safety, safetyZ, &safety);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_