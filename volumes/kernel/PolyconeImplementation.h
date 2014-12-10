/*
 * PolyconeImplementation.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/kernel/ConeImplementation.h"

#include <cassert>
#include <stdio.h>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(PolyconeImplementation,
        TranslationCode,transCodeT, RotationCode,rotCodeT)

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolycone;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct PolyconeImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedPolycone;
  using UnplacedShape_t = UnplacedPolycone;

  // here put all the implementations
  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
      printf("SpecializedPolycone<%i, %i>", transCodeT, rotCodeT);
  }

  /////GenericKernel Contains/Inside implementation
  template <typename Backend, bool ForInside>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedPolycone const &polycone,
            Vector3D<typename Backend::precision_v> const &point,
            typename Backend::bool_v &completelyinside,
            typename Backend::bool_v &completelyoutside)
     {
        // TBD
     }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void ContainsKernel(
        UnplacedPolycone const &polycone,
        Vector3D<typename Backend::precision_v> const &point,
        typename Backend::bool_v &inside)
    {
        // add z check
        if( point.z() < polycone.fZs[0] || point.z() > polycone.fZs[polycone.GetNSections()] )
        {
            inside = Backend::kFalse;
            return;
        }

        // now we have to find a section
        PolyconeSection const & sec = polycone.GetSection(point.z());

        Vector3D<Precision> localp;
        ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::Contains<Backend>(
                *sec.solid,
                Transformation3D::kIdentity,
                point - Vector3D<Precision>(0,0,sec.shift),
                localp,
                inside
        );
        return;
    }

    //template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    static void InsideKernel(
       UnplacedPolycone const &polycone,
       Vector3D<typename Backend::precision_v> const &point,
       typename Backend::inside_v &inside) {

        typename Backend::bool_v contains;
        ContainsKernel<Backend>(polycone,point,contains);
        if(contains){
            inside = EInside::kInside;
        }
        else
        {
            inside = EInside::kOutside;
        }
    }


    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void UnplacedContains( UnplacedPolycone const &polycone,
        Vector3D<typename Backend::precision_v> const &point,
        typename Backend::bool_v &inside) {
      // TODO: do this generically WITH a generic contains/inside kernel
      // forget about sector for the moment
      ContainsKernel<Backend>(polycone, point, inside);
    }

    template <typename Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Contains(
        UnplacedPolycone const &unplaced,
        Transformation3D const &transformation,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> &localPoint,
        typename Backend::bool_v &inside){

        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedContains<Backend>(unplaced, localPoint, inside);
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void Inside(UnplacedPolycone const &polycone,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       typename Backend::inside_v &inside) {

      InsideKernel<Backend>(polycone, transformation.Transform<transCodeT, rotCodeT>(point),  inside);

  }


      template <class Backend>
      VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      static void DistanceToIn(
          UnplacedPolycone const &polycone,
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
        UnplacedPolycone const &polycone,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &dir,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void SafetyToIn(UnplacedPolycone const &polycone,
                           Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point,
                           typename Backend::precision_v &safety) {


    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void SafetyToOut(UnplacedPolycone const &polycone,
                            Vector3D<typename Backend::precision_v> const &point,
                            typename Backend::precision_v &safety) {


    }

}; // end PolyconeImplementation

}} // end namespace


#endif /* POLYCONEIMPLEMENTATION_H_ */
