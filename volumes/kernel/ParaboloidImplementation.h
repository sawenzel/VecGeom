/// @file ParaboloidImplementation.h

#ifndef VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_

#include "base/global.h"

#include "base/transformation3d.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedParaboloid.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParaboloidImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedInside(
      UnplacedParaboloid const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      typename Backend::int_v &inside) {
    // NYI
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedParaboloid const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> &localPoint,
                     typename Backend::int_v &inside) {
    // NYI
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(
      UnplacedParaboloid const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {
    // NYI
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(
      UnplacedParaboloid const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {
    // NYI
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedParaboloid const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
    // NYI
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedParaboloid const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {
    // NYI
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_