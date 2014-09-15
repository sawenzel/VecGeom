/// @file TorusImplementation.h

#ifndef VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_


#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTorus.h"

namespace VECGEOM_NAMESPACE {


template <TranslationCode transCodeT, RotationCode rotCodeT>
struct TorusImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains( UnplacedTorus const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {

    typedef typename Backend::precision_v Float_t;

    // TODO: do this generically WITH a generic contains/inside kernel
    // forget about sector for the moment

    Float_t rxy = Sqrt(point[0]*point[0] + point[1]*point[1]);
    Float_t radsq = ( rxy - torus.rtor() ) * (rxy - torus.rtor() ) + point[2]*point[2];
    inside = radsq > torus.rmin2() && radsq < torus.rmax2();
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedTorus const &torus,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(torus, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTorus const &torus,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside) {

    // TODO
  }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void DistanceToIn(
        UnplacedTorus const &torus,
        Transformation3D const &transformation,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &direction,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {
      
    // TODO
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTorus const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    distance = kInfinity;

    // TODO
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTorus const &torus,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

    // TODO
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTorus const &torus,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {

    // TODO

}

}; // end struct

} // end namespace


#endif // VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_
