/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANINTERSECTIONIMPLEMENTATION_H_
#define BOOLEANINTERSECTIONIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBooleanVolume.h"

namespace VECGEOM_NAMESPACE {

/**
 * partial template specialization for UNION implementation
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanImplementation<kIntersection, transCodeT, rotCodeT> {

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

		// copied from Geant4
	    double dist = 0.0;
		/*
	    if( Inside(p) == kInside )
	    {
	  #ifdef G4BOOLDEBUG
	      G4cout << "WARNING - Invalid call in "
	             << "G4IntersectionSolid::DistanceToIn(p,v)" << G4endl
	             << "  Point p is inside !" << G4endl;
	      G4cout << "          p = " << p << G4endl;
	      G4cout << "          v = " << v << G4endl;
	      G4cerr << "WARNING - Invalid call in "
	             << "G4IntersectionSolid::DistanceToIn(p,v)" << G4endl
	             << "  Point p is inside !" << G4endl;
	      G4cerr << "          p = " << p << G4endl;
	      G4cerr << "          v = " << v << G4endl;
	  #endif
	    }*/
		
	    VPlacedVolume const*const fPtrSolidA = unplaced.fLeftVolume;
	    VPlacedVolume const*const fPtrSolidB = unplaced.fRightVolume;
	    typename Backend::inside_v wA = fPtrSolidA->Inside(p);
	    typename Backend::inside_v wB = fPtrSolidB->Inside(p);

	    Vector3D<Precision> pA = p,  pB = p;
	    double      dA = 0., dA1=0., dA2=0.;
	    double      dB = 0., dB1=0., dB2=0.;
	    bool        doA = true, doB = true;

	    while(true) {
	        if(doA) 
	        {
	          // find next valid range for A
	          dA1 = 0.;

	          if( wA != EInside::kInside ) 
	          {
	            dA1 = fPtrSolidA->DistanceToIn(pA, v);
	            if( dA1 == kInfinity ){
					distance = kInfinity;
					return;
				   }
	            pA += dA1*v;
	          }
	          dA2 = dA1 + fPtrSolidA->DistanceToOut(pA, v);
	        }
	        dA1 += dA;
	        dA2 += dA;

	        if(doB) 
	        {
	          // find next valid range for B

	          dB1 = 0.;
	          if(wB != EInside::kInside) 
	          {
	            dB1 = fPtrSolidB->DistanceToIn(pB, v);
	            if(dB1 == kInfinity){  
					distance = kInfinity;
					return;
				}
	            pB += dB1*v;
	          }
	          dB2 = dB1 + fPtrSolidB->DistanceToOut(pB, v);
	        }
	        dB1 += dB;
	        dB2 += dB;

	         // check if they overlap
	        if( dA1 < dB1 ) 
	        {
	          if( dB1 < dA2 ){
				  distance = dB1;
				    return;
				}
	          dA   = dA2;
	          pA   = p + dA*v;  // continue from here
	          wA   = EInside::kSurface;
	          doA  = true;
	          doB  = false;
	        }
	        else 
	        {
	          if( dA1 < dB2 ){  
				  distance = dA1;
				  return;
			  }
	          dB   = dB2;
	          pB   = p + dB*v;  // continue from here
	          wB   = EInside::kSurface;
	          doB  = true;
	          doA  = false;
	        }
	      }
  		distance = dist;
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
	               unplaced.fRightVolume->DistanceToOut(p,v));
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kIntersection, transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety) {

		typedef typename Backend::bool_v Bool_t;

	Bool_t insideA = unplaced.fLeftVolume->Contains(p) ;
	Bool_t insideB = unplaced.fRightVolume->Contains(p) ;
	
	    if( ! insideA && insideB  )
	    {
	      safety = unplaced.fLeftVolume->SafetyToIn(p) ;
	    }
	    else
	    {
	      if( ! insideB  && insideA  )
	      {
	        safety = unplaced.fRightVolume->SafetyToIn(p) ;
	      }
	      else
	      {
	        safety =  Min(unplaced.fLeftVolume->SafetyToIn(p),
	                      unplaced.fRightVolume->SafetyToIn(p) ) ; 
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

	safety = Min(unplaced.fLeftVolume->SafetyToOut(p),
	 unplaced.fRightVolume->SafetyToOut(p));
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

} // End global namespace



#endif /* BooleanImplementation_H_ */
