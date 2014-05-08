/// @file ParallelepipedImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_

#include "base/global.h"

#include "base/transformation3d.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedParallelepiped.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParallelepipedImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void Transform(UnplacedParallelepiped const &unplaced,
                        Vector3D<typename Backend::precision_v> &point) {
    point[1] -= unplaced.GetTanThetaSinPhi()*point[2];
    point[0] -= unplaced.GetTanThetaCosPhi()*point[2]
                + unplaced.GetTanAlpha()*point[1];
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedInside(
      UnplacedParallelepiped const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      typename Backend::int_v &inside) {

    Transform<Backend>(unplaced, point);

    // Run unplaced box kernel
    BoxImplementation<transCodeT, rotCodeT>::template
        InsideKernel<Backend>(unplaced.GetDimensions(), point, inside);

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedParallelepiped const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> &localPoint,
                     typename Backend::int_v &inside) {
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedInside<Backend>(unplaced, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(
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

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(
      UnplacedParallelepiped const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    Transform<Backend>(unplaced, point);
    Transform<Backend>(unplaced, direction);

    // Run unplaced box kernel
    BoxImplementation<transCodeT, rotCodeT>::template
        DistanceToOutKernel<Backend>(unplaced.GetDimensions(), point,
                                     direction, stepMax, distance);

  }


  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedParallelepiped const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

    typedef typename Backend::precision_v Float_t;

    Vector3D<Float_t> localPoint =
        transformation.Transform<transCodeT, rotCodeT>(point);
    Transform<Backend>(unplaced, localPoint);

    // Run unplaced box kernel
    BoxImplementation<transCodeT, rotCodeT>::template
        SafetyToInKernel<Backend>(unplaced.GetDimensions(), localPoint, safety);
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedParallelepiped const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {

    Transform<Backend>(unplaced, point);

    // Run unplaced box kernel
    BoxImplementation<transCodeT, rotCodeT>::template
        SafetyToOutKernel<Backend>(unplaced.GetDimensions(), point, safety);
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_