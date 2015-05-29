/// @file ParallelepipedImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedParallelepiped.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(ParallelepipedImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedParallelepiped;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParallelepipedImplementation {

#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL int transC = transCodeT;
  VECGEOM_GLOBAL int rotC   = rotCodeT;
#else
  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;
#endif

  using PlacedShape_t = PlacedParallelepiped;
  using UnplacedShape_t = UnplacedParallelepiped;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedParallelepiped<%i, %i>", transCodeT, rotCodeT);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Transform(UnplacedParallelepiped const &unplaced,
                        Vector3D<typename Backend::precision_v> &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedParallelepiped const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedParallelepiped const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedParallelepiped const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedParallelepiped const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedParallelepiped const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedParallelepiped const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedParallelepiped const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

};

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::Transform(
    UnplacedParallelepiped const &unplaced,
    Vector3D<typename Backend::precision_v> &point) {
  point[1] -= unplaced.GetTanThetaSinPhi()*point[2];
  point[0] -= unplaced.GetTanThetaCosPhi()*point[2]
              + unplaced.GetTanAlpha()*point[1];
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedParallelepiped const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::bool_v &inside) {

  Vector3D<typename Backend::precision_v> localPoint = point;
  Transform<Backend>(unplaced, localPoint);

  // Run unplaced box kernel
  BoxImplementation<transCodeT, rotCodeT>::template
      ContainsKernel<Backend>(unplaced.GetDimensions(), localPoint, inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedParallelepiped const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {
  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedParallelepiped const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  Vector3D<typename Backend::precision_v> localPoint =
      transformation.Transform<transCodeT, rotCodeT>(point);
  Transform<Backend>(unplaced, localPoint);
  BoxImplementation<transCodeT, rotCodeT>::template
      InsideKernel<Backend>(unplaced.GetDimensions(), localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedParallelepiped const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;

  Vector3D<Float_t> localPoint =
      transformation.Transform<transCodeT, rotCodeT>(point);
  Vector3D<Float_t> localDirection =
      transformation.TransformDirection<rotCodeT>(direction);

  Transform<Backend>(unplaced, localPoint);
  Transform<Backend>(unplaced, localDirection);

  // Run unplaced box kernel
  BoxImplementation<transCodeT, rotCodeT>::template
      DistanceToInKernel<Backend>(unplaced.GetDimensions(), localPoint,
                                  localDirection, stepMax, distance);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedParallelepiped const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t max;
  Bool_t inPoint, inDirection, goingAway;
  Bool_t done = Backend::kFalse;
  distance = kInfinity;

  // Z intersection

  inDirection = direction[2] > 0;
  max = unplaced.GetZ() - point[2];
  inPoint = max > kHalfTolerance;
  goingAway = inDirection && !inPoint;
  MaskedAssign(goingAway, 0., &distance);
  MaskedAssign(inPoint && inDirection, max/direction[2], &distance);
  done |= goingAway;
  if (done == Backend::kTrue) return;

  inDirection = direction[2] < 0;
  max = -unplaced.GetZ() - point[2];
  inPoint = max < -kHalfTolerance;
  goingAway = inDirection && !inPoint;
  MaskedAssign(goingAway, 0., &distance);
  MaskedAssign(inPoint && inDirection, max/direction[2], &distance);
  done |= goingAway;
  if (done == Backend::kTrue) return;
    
  // Y plane intersection

  Float_t localPointY, localDirectionY;

  localPointY = point[1] - unplaced.GetTanThetaSinPhi()*point[2];
  localDirectionY = direction[1] - unplaced.GetTanThetaSinPhi()*direction[2];

  inDirection = localDirectionY > 0;
  max = unplaced.GetY() - localPointY;
  inPoint = max > kHalfTolerance;
  goingAway = inDirection && !inPoint;
  max /= localDirectionY;
  MaskedAssign(goingAway, 0., &distance);
  MaskedAssign(inPoint && inDirection && max < distance, max, &distance);
  done |= goingAway;
  if (done == Backend::kTrue) return;

  inDirection = localDirectionY < 0;
  max = -unplaced.GetY() - localPointY;
  inPoint = max < -kHalfTolerance;
  goingAway = inDirection && !inPoint;
  max /= localDirectionY;
  MaskedAssign(goingAway, 0., &distance);
  MaskedAssign(inPoint && inDirection && max < distance, max, &distance);
  done |= goingAway;
  if (done == Backend::kTrue) return;

  // X plane intersection

  Float_t localPointX, localDirectionX;

  localPointX = point[0] - unplaced.GetTanThetaCosPhi()*point[2]
                         - unplaced.GetTanAlpha()*localPointY;
  localDirectionX = direction[0] - unplaced.GetTanThetaCosPhi()*direction[2]
                                 - unplaced.GetTanAlpha()*localDirectionY;

  inDirection = localDirectionX > 0;
  max = unplaced.GetX() - localPointX;
  inPoint = max > kHalfTolerance;
  goingAway = inDirection && !inPoint;
  max /= localDirectionX;
  MaskedAssign(goingAway, 0., &distance);
  MaskedAssign(inPoint && inDirection && max < distance, max, &distance);
  done |= goingAway;
  if (done == Backend::kTrue) return;

  inDirection = localDirectionX < 0;
  max = -unplaced.GetX() - localPointX;
  inPoint = max < -kHalfTolerance;
  goingAway = inDirection && !inPoint;
  max /= localDirectionX;
  MaskedAssign(goingAway, 0., &distance);
  MaskedAssign(inPoint && inDirection && max < distance, max, &distance);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedParallelepiped const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  Float_t cty, ctx;
  Vector3D<Float_t> safetyVector;

  Vector3D<Float_t> localPoint =
      transformation.Transform<transCodeT, rotCodeT>(point);
  Transform<Backend>(unplaced, localPoint);

  safetyVector[0] = Abs(localPoint[0]) - unplaced.GetX();
  safetyVector[1] = Abs(localPoint[1]) - unplaced.GetY();
  safetyVector[2] = Abs(localPoint[2]) - unplaced.GetZ();

  ctx = 1.0 / Sqrt(1. + unplaced.GetTanAlpha()*unplaced.GetTanAlpha()
                      + unplaced.GetTanThetaCosPhi()
                        *unplaced.GetTanThetaCosPhi());

  cty = 1.0 / Sqrt(1. + unplaced.GetTanThetaSinPhi()
                        *unplaced.GetTanThetaSinPhi());

  safetyVector[0] *= ctx;
  safetyVector[1] *= cty;

  safety = safetyVector[0];
  MaskedAssign(safetyVector[1] > safety, safetyVector[1], &safety);
  MaskedAssign(safetyVector[2] > safety, safetyVector[2], &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedParallelepiped const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  Float_t cty, ctx;
  Vector3D<Float_t> safetyVector;
  Vector3D<Float_t> localPoint = point;
  Transform<Backend>(unplaced, localPoint);

  safetyVector[0] = unplaced.GetX() - Abs(localPoint[0]);
  safetyVector[1] = unplaced.GetY() - Abs(localPoint[1]);
  safetyVector[2] = unplaced.GetZ() - Abs(localPoint[2]);

  ctx = 1.0 / Sqrt(1. + unplaced.GetTanAlpha()*unplaced.GetTanAlpha()
                      + unplaced.GetTanThetaCosPhi()
                        *unplaced.GetTanThetaCosPhi());

  cty = 1.0 / Sqrt(1. + unplaced.GetTanThetaSinPhi()
                        *unplaced.GetTanThetaSinPhi());

  safetyVector[0] *= ctx;
  safetyVector[1] *= cty;

  safety = safetyVector[0];
  MaskedAssign(safetyVector[1] < safety, safetyVector[1], &safety);
  MaskedAssign(safetyVector[2] < safety, safetyVector[2], &safety);
}

} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
