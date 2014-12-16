/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANINTERSECTIONIMPLEMENTATION_H_
#define BOOLEANINTERSECTIONIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBooleanVolume.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * partial template specialization for UNION implementation
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanImplementation<kIntersection, transCodeT, rotCodeT> {
  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedBooleanVolume;
  using UnplacedShape_t = UnplacedBooleanVolume;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedBooleanVolume<%i, %i, %i>", kIntersection, transCodeT, rotCodeT);
  }

  //
  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedBooleanVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedBooleanVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedBooleanVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedBooleanVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedBooleanVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedBooleanVolume const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedBooleanVolume const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      UnplacedBooleanVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      UnplacedBooleanVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedBooleanVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      UnplacedBooleanVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      UnplacedBooleanVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      UnplacedBooleanVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedBooleanVolume const & unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );

}; // End struct BooleanImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>( unplaced , localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::Contains(
        UnplacedBooleanVolume const &unplaced,
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
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::Inside(
    UnplacedBooleanVolume const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  InsideKernel<Backend>(unplaced,
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::DistanceToIn(
    UnplacedBooleanVolume const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToInKernel<Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    transformation.TransformDirection<rotCodeT>(direction),
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::DistanceToOut(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Backend>(
    unplaced,
    point,
    direction,
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::SafetyToIn(
    UnplacedBooleanVolume const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToInKernel<Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::SafetyToOut(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToOutKernel<Backend>(
    unplaced,
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::ContainsKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

	typedef typename Backend::bool_v Bool_t;
	Bool_t insideA = unplaced.fLeftVolume->Contains(localPoint);
	Bool_t insideB = unplaced.fRightVolume->Contains(localPoint);
	inside = insideA && insideB;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::InsideKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::inside_v &inside) {

  // now use the Inside functionality of left and right components
  // algorithm taken from Geant4 implementation
  VPlacedVolume const*const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const*const fPtrSolidB = unplaced.fRightVolume;

  typename Backend::inside_v positionA = fPtrSolidA->Inside(p) ;

  if( positionA == EInside::kOutside ){
	  inside = EInside::kOutside;
	return;
  } 

  typename Backend::inside_v positionB = fPtrSolidB->Inside(p) ;
  
  if(positionA == EInside::kInside 
	  && positionB == EInside::kInside)
  {
	  inside=EInside::kInside;
    return;
  }
  else
  {
    if((positionA == EInside::kInside && positionB == EInside::kSurface) ||
       (positionB == EInside::kInside && positionA == EInside::kSurface) ||
       (positionA == EInside::kSurface && positionB == EInside::kSurface)   )
    {
		inside = EInside::kSurface;
	  return;
    }
    else
    {
		inside=EInside::kOutside;
	  return;
    }
  }
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Vector3D<Precision> hitpoint = p;

    Bool_t inleft = unplaced.fLeftVolume->Contains( hitpoint );
    Bool_t inright = unplaced.fRightVolume->Contains( hitpoint );
    Float_t d1 = 0.;
    Float_t d2 = 0.;
    Float_t snext = 0.0;

    // just a pre-check before entering main algorithm
    if (inleft && inright) {
          d1 = unplaced.fLeftVolume->PlacedDistanceToOut( hitpoint, v, stepMax);
          d2 = unplaced.fRightVolume->PlacedDistanceToOut( hitpoint, v, stepMax);

          // if we are close to a boundary continue
          if (d1<2*kTolerance) inleft = Backend::kFalse;
          if (d2<2*kTolerance) inright = Backend::kFalse;

          // otherwise exit
          if (inleft && inright){
              // TODO: WE are inside both so should return a negative number
              distance = 0.0;
              return; }
    }

    // main loop
    while (1) {
          d1 = d2 = 0;
          if (!inleft)  {
             d1 = unplaced.fLeftVolume->DistanceToIn( hitpoint, v );
             d1 = Max( d1, kTolerance );
             if ( d1 >1E20 ){ distance = kInfinity; return; }
          }
          if (!inright) {
             d2 = unplaced.fRightVolume->DistanceToIn( hitpoint, v );
             d2 = Max( d2, kTolerance );
             if ( d2>1E20 ){ distance = kInfinity; return; }
          }

          if (d1>d2) {
             // propagate to left shape
             snext += d1;
             inleft = Backend::kTrue;
             hitpoint += d1*v;

             // check if propagated point is inside right shape
             // check is done with a little push
             inright = unplaced.fRightVolume->Contains( hitpoint + kTolerance*v );
             if (inright){
                 distance = snext;
                 return;}
             // here inleft=true, inright=false
          } else {
             // propagate to right shape
             snext += d2;
             inright = Backend::kTrue;
             hitpoint += d2*v;

             // check if propagated point is inside left shape
             inleft = unplaced.fLeftVolume->Contains(hitpoint + kTolerance*v );
             if (inleft){
                distance = snext;
                return;
             }
          }
             // here inleft=false, inright=true
     } // end while loop
     distance = snext;
     return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

	distance = Min(unplaced.fLeftVolume->DistanceToOut(p,v),
	               unplaced.fRightVolume->PlacedDistanceToOut(p,v));
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety) {

    typedef typename Backend::bool_v Bool_t;

    // This is the Geant4 algorithm
    // TODO: ROOT seems to produce better safeties

    Bool_t insideA = unplaced.fLeftVolume->Contains(p);
    Bool_t insideB = unplaced.fRightVolume->Contains(p);

    if( ! insideA && insideB  )
    {
       safety = unplaced.fLeftVolume->SafetyToIn(p);
    }
    else
    {
       if( ! insideB  && insideA  )
       {
          safety = unplaced.fRightVolume->SafetyToIn(p);
       }
       else
       {
           safety =  Min(unplaced.fLeftVolume->SafetyToIn(p),
                         unplaced.fRightVolume->SafetyToIn(p));
       }
    }
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety) {

    safety = Min(
             // TODO: could fail if left volume is placed shape
             unplaced.fLeftVolume->SafetyToOut(p),

             // TODO: consider introducing PlacedSafetyToOut function
             unplaced.fRightVolume->SafetyToOut(
                     unplaced.fRightVolume->transformation()->Transform(p))
             );
    MaskedAssign( safety < 0, 0., &safety);
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::NormalKernel(
     UnplacedBooleanVolume const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {
    // TBDONE
}

} // End impl namespace

} // End global namespace



#endif /* BooleanImplementation_H_ */
