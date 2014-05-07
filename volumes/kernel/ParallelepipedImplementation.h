/// @file ParallelepipedKernel.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDKERNEL_H_
#define VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDKERNEL_H_

#include "base/global.h"

#include "base/transformation3d.h"
#include "volumes/kernel/KernelUtilities.h"
#include "volumes/UnplacedParallelepiped.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParallelepipedImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedInside(
      UnplacedParallelepiped const &unplaced,
      Vector3D<typename Backend::precision_v> localPoint,
      typename Backend::int_v &inside) {

    typedef typename Backend::precision_v Float_t;

    Float_t shift = unplaced.GetTanThetaSinPhi()*localPoint[2];
    localPoint[0] -= shift + unplaced.GetTanAlpha()*localPoint[1];
    localPoint[1] -= shift;

    // Run regular kernel for point inside a box
    KernelUtilities<Backend>::LocalPointInsideBox(
      unplaced.GetDimensions(), localPoint, inside
    );

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
      Vector3D<typename Backend::precision_v> const &pos,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &step_max,
      typename Backend::precision_v &distance) {
    // NYI
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(
      UnplacedParallelepiped const &unplaced,
      Vector3D<typename Backend::precision_v> const &pos,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &step_max,
      typename Backend::precision_v &distance) {
    // NYI
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

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDKERNEL_H_