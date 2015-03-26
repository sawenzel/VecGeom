/// @file BoxImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBox.h"
#include "volumes/kernel/GenericKernels.h"

#include <stdio.h>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(BoxImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)


inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox;
class UnplacedBox;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BoxImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedBox;
  using UnplacedShape_t = UnplacedBox;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
  }

  template<typename Backend>
  VECGEOM_INLINE
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

  template <typename Backend, bool ForInside>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(Vector3D<Precision> const &,
          Vector3D<typename Backend::precision_v> const &,
          typename Backend::bool_v &,
          typename Backend::bool_v &);

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

 // template <class Backend>
 // static void Normal( Vector3D<Precision> const &dimensions,
 //         Vector3D<typename Backend::precision_v> const &point,
 //        Vector3D<typename Backend::precision_v> &normal,
 //         Vector3D<typename Backend::precision_v> &valid )

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedBox const &,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );

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

  typedef typename Backend::bool_v Bool_t;
  Bool_t unused;
  Bool_t outside;
  GenericKernelForContainsAndInside<Backend, false>(dimensions,
    localPoint, unused, outside);
  inside=!outside;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

//    using vecgeom::GenericKernels;
// here we are explicitely unrolling the loop since  a for statement will likely be a penality
// check if second call to Abs is compiled away
    // and it can anyway not be vectorized
    /* x */
    completelyoutside = Abs(localPoint[0]) > MakePlusTolerant<ForInside>( dimensions[0] );
    if (ForInside)
    {
        completelyinside = Abs(localPoint[0]) < MakeMinusTolerant<ForInside>( dimensions[0] );
    }
    if (Backend::early_returns) {
      if ( IsFull (completelyoutside) ) {
        return;
      }
    }
/* y */
    completelyoutside |= Abs(localPoint[1]) > MakePlusTolerant<ForInside>( dimensions[1] );
    if (ForInside)
    {
      completelyinside &= Abs(localPoint[1]) < MakeMinusTolerant<ForInside>( dimensions[1] );
    }
    if (Backend::early_returns) {
      if ( IsFull (completelyoutside) ) {
        return;
      }
    }
/* z */
    completelyoutside |= Abs(localPoint[2]) > MakePlusTolerant<ForInside>( dimensions[2] );
    if (ForInside)
    {
      completelyinside &= Abs(localPoint[2]) < MakeMinusTolerant<ForInside>( dimensions[2] );
    }
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::InsideKernel(
    Vector3D<Precision> const &boxDimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  typedef typename Backend::bool_v      Bool_t;
  Bool_t completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Backend,true>(
      boxDimensions, point, completelyinside, completelyoutside);
  inside=EInside::kSurface;
  MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  MaskedAssign(completelyinside, EInside::kInside, &inside);
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

#ifdef VECGEOM_NVCC
  #define surfacetolerant true
#else
  static const bool surfacetolerant=true;
#endif

  safety[0] = Abs(point[0]) - dimensions[0];
  safety[1] = Abs(point[1]) - dimensions[1];
  safety[2] = Abs(point[2]) - dimensions[2];

  done |= (safety[0] >= stepMax ||
           safety[1] >= stepMax ||
           safety[2] >= stepMax);
  if ( IsFull(done) ) return;

  Float_t next, coord1, coord2;
  Bool_t hit;

  // x
  next = safety[0] / Abs(direction[0] + kMinimum);
  coord1 = point[1] + next * direction[1];
  coord2 = point[2] + next * direction[2];
  hit = safety[0] >= MakeMinusTolerant<surfacetolerant>(0.) &&
        point[0] * direction[0] < 0 &&
        Abs(coord1) <= dimensions[1] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, &distance);
  done |= hit;
  if ( IsFull(done) ) return;

  // y
  next = safety[1] / Abs(direction[1] + kMinimum);
  coord1 = point[0] + next * direction[0];
  coord2 = point[2] + next * direction[2];
  hit = safety[1] >= MakeMinusTolerant<surfacetolerant>(0.) &&
        point[1] * direction[1] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, &distance);
  done |= hit;
  if ( IsFull(done) ) return;

  // z
  next = safety[2] / Abs(direction[2] + kMinimum);
  coord1 = point[0] + next * direction[0];
  coord2 = point[1] + next * direction[1];
  hit = safety[2] >= MakeMinusTolerant<surfacetolerant>(0.) &&
        point[2] * direction[2] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[1];
  MaskedAssign(!done && hit, next, &distance);

#ifdef VECGEOM_NVCC
  #undef surfacetolerant
#endif
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
    // typedef typename Backend::bool_v Bool_t;

    Vector3D<Float_t> safety;
    // Bool_t inside;

    distance = kInfinity;

    //safety[0] = Abs(point[0]) - dimensions[0];
    //safety[1] = Abs(point[1]) - dimensions[1];
    //safety[2] = Abs(point[2]) - dimensions[2];

    //inside = safety[0] < stepMax &&
    //         safety[1] < stepMax &&
    //         safety[2] < stepMax;
    //if (inside == Backend::kFalse) return;

    Vector3D<Float_t> inverseDirection = Vector3D<Float_t>(
      1. / (direction[0] + kMinimum),
      1. / (direction[1] + kMinimum),
      1. / (direction[2] + kMinimum)
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

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::NormalKernel(
     UnplacedBox const &box,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {

        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;

         // Computes the normal on a surface and returns it as a unit vector
         //   In case a point is further than tolerance_normal from a surface, set validNormal=false
         //   Must return a valid vector. (even if the point is not on the surface.)
         //
         //   On an edge or corner, provide an average normal of all facets within tolerance
         // NOTE: the tolerance value used in here is not yet the global surface
         //     tolerance - we will have to revise this value - TODO
         // this version does not yet consider the case when we are not on the surface

         Vector3D<Precision> dimensions= box.dimensions();

         constexpr double delta = 100.*kTolerance;
         constexpr double kInvSqrt2 = 0.7071067811865475; // = 1. / Sqrt(2.);
         constexpr double kInvSqrt3 = 0.5773502691896258; // = 1. / Sqrt(3.);
         normal.Set(0.);
         Float_t nsurf = 0;
         Float_t safmin(kInfinity);

         // do a loop here over dimensions
         for( int dim = 0; dim < 3; ++dim )
         {
             Float_t currentsafe = Abs(Abs(point[dim]) - dimensions[dim]);
             MaskedAssign( currentsafe < safmin, currentsafe, &safmin );

             // close to this surface
             Bool_t closetoplane = currentsafe < delta;
             if( ! IsEmpty( closetoplane ) )
             {
                Float_t nsurftmp = nsurf + 1.;

                Float_t sign(1.);
                MaskedAssign( point[dim] < 0, -1., &sign);
                Float_t tmpnormalcomponent = normal[dim] + sign;

                MaskedAssign( closetoplane, nsurftmp, &nsurf );
                MaskedAssign( closetoplane, tmpnormalcomponent, &normal[dim] );
             }
         }

         valid = Backend::kTrue;
         valid &= nsurf>0;
         MaskedAssign( nsurf == 3., normal*kInvSqrt3, &normal );
         MaskedAssign( nsurf == 2., normal*kInvSqrt2, &normal );

        // TODO: return normal in case of nonvalid case;
        // need to keep track of minimum safety direction
    }


} } // End global namespace


#endif // VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
