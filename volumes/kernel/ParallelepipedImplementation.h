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

    // Run regular unplaced kernel for point inside a box
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
    localPoint = transformation.template Transform<transCodeT, rotCodeT>(point);
    UnplacedInside<Backend>(unplaced, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(
      UnplacedParallelepiped const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    point = transformation.Transform(point);
    direction = transformation.TransformDirection(direction);

    Transform<Backend>(unplaced, point);
    Transform<Backend>(unplaced, direction);

    // Run regular unplaced kernel for distance to a box
    BoxImplementation<transCodeT, rotCodeT>::template
        DistanceToInKernel<Backend>(unplaced.GetDimensions(), point, direction,
                                    stepMax, distance
    );

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

    // Run regular unplaced kernel for distance to a box
    BoxImplementation<transCodeT, rotCodeT>::template
        DistanceToOutKernel<Backend>(unplaced.GetDimensions(), point,
                                     direction, stepMax, distance
    );

  }


  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedParallelepiped const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
    // NYI
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedParallelepiped const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {
    // NYI
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_