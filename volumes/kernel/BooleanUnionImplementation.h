/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANUNIONIMPLEMENTATION_H_
#define BOOLEANUNIONIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBooleanVolume.h"

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * partial template specialization for UNION implementation
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanImplementation<kUnion, transCodeT, rotCodeT> {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedBooleanVolume;
  using UnplacedShape_t = UnplacedBooleanVolume;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedBooleanVolume<%i, %i, %i>", kUnion, transCodeT, rotCodeT);
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>( unplaced , localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::Contains(
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::Inside(
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToIn(
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToOut(
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToIn(
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToOut(
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
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::ContainsKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

   inside = unplaced.fLeftVolume->Contains(localPoint);
   if ( IsFull(inside) ) return;
   inside |= unplaced.fRightVolume->Contains(localPoint);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::InsideKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::inside_v &inside) {

  // now use the Inside functionality of left and right components
  // algorithm taken from Geant4 implementation
  VPlacedVolume const*const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const*const fPtrSolidB = unplaced.fRightVolume;

  typename Backend::inside_v positionA = fPtrSolidA->Inside(p);
  if (positionA == EInside::kInside){
        inside = EInside::kInside;
      return;
  }

  typename Backend::inside_v positionB = fPtrSolidB->Inside(p);
  if( positionB == EInside::kInside
      /* leaving away this part of the condition for the moment until SurfaceNormal implemented
      ||
    ( positionA == EInside::kSurface && positionB == EInside::kSurface )


    &&
        ( fPtrSolidA->SurfaceNormal(p) +
          fPtrSolidB->SurfaceNormal(p) ).mag2() <
          1000*G4GeometryTolerance::GetInstance()->GetRadialTolerance() )
      */ )
  {
    inside = EInside::kInside;
    return;
  }
  else
  {
    if( ( positionB == EInside::kSurface ) || ( positionA == EInside::kSurface ) ){
        inside = EInside::kSurface;
        return;
    }
    else {
        inside = EInside::kOutside;
        return;
    }
  }
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;

  Float_t d1 = unplaced.fLeftVolume->DistanceToIn( p, v, stepMax );
  Float_t d2 = unplaced.fRightVolume->DistanceToIn( p, v, stepMax );
  distance = Min(d1,d2);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    Float_t dist = 0., disTmp = 0.;
    int push=0;
    // std::cout << "##VECGEOMSTART\n";
    typename Backend::Bool_t positionA = fPtrSolidA->Contains(p);
    if( positionA )
    {
      do
       {
         //count1++;
	    // we don't need a transformation here
         disTmp = fPtrSolidA->DistanceToOut(p+dist*v,v);
         // distTmp
	    //   std::cout << "VecdistTmp1 " << disTmp << "\n";
         dist += ( disTmp >= 0. && disTmp < kInfinity )? disTmp : 0;
	    // give a push
	    dist += kTolerance;
	 push++;

         if( fPtrSolidB->Contains(p+dist*v) )
         {
            disTmp = fPtrSolidB->PlacedDistanceToOut(p+dist*v,v);
	    // std::cout << "VecdistTmp2 " << disTmp << "\n";
	    dist += ( disTmp >= 0. && disTmp < kInfinity )? disTmp : 0;
             //dist += (disTmp>=0.)? disTmp : 0.;
         }
        //if(count1 > 100){
            //std::cerr << "LOOP1 INFINITY\n"; break; }
       }
       while( fPtrSolidA->Contains(p+dist*v) );
      // NOTE was kCarTolerance; just taking kHalfHolerance
    }
    else // if( positionB != kOutside )
    {
      do
      {
	    disTmp = fPtrSolidB->PlacedDistanceToOut(p+dist*v,v);
    	//  std::cout << "VecdistTmp3 " << disTmp << "\n";
        //dist += disTmp ;
        dist += (disTmp>=0. && disTmp<kInfinity)? disTmp : 0.;
	    dist += kTolerance;
	    push++;
        if( fPtrSolidA->Contains(p+dist*v) )
        {
             disTmp = fPtrSolidA->DistanceToOut(p+dist*v,v);
             //std::cerr << "distTmp4 " << disTmp;
	     //   std::cout << "VecdistTmp4 " << disTmp << "\n";
             dist += disTmp ;
	     //dist += (disTmp>=0.)? disTmp : 0.;
        }

	//        if(count2 > 100){
        //     std::cerr << "LOOP2 INFINITY\n"; break; }
      }
      while(fPtrSolidB->Contains(p+dist*v) );
    }
    //  std::cerr << "--VecGeom return " << dist << "\n";
    distance = dist - push*kTolerance;
    return;
}


/* template <TranslationCode transCodeT, RotationCode rotCodeT> */
/* template <typename Backend> */
/* VECGEOM_CUDA_HEADER_BOTH */
/* void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToOutKernel( */
/*     UnplacedBooleanVolume const & unplaced, */
/*     Vector3D<typename Backend::precision_v> const &p, */
/*     Vector3D<typename Backend::precision_v> const &v, */
/*     typename Backend::precision_v const &stepMax, */
/*     typename Backend::precision_v &distance) { */

/*     /\* algorithm taken from Geant4 *\/ */

/*     typedef typename Backend::precision_v Float_t; */
/*     VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume; */
/*     VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume; */

/*     Float_t dist = 0., disTmp = 0.; */
/*     int count1=0, count2=0; */
/*     // std::cout << "##VECGEOMSTART\n"; */
/*     typename Backend::inside_v positionA = fPtrSolidA->Inside(p); */
/*     if( positionA != EInside::kOutside ) */
/*     { */
/*       do */
/*        { */
/*          //count1++; */
/*          disTmp = fPtrSolidA->DistanceToOut(p+dist*v,v); */
/*          // distTmp */
/* 	 //   std::cout << "VecdistTmp1 " << disTmp << "\n"; */

/*          dist += disTmp; */

/*          if(fPtrSolidB->Inside(p+dist*v) != EInside::kOutside) */
/*          { */
/*             disTmp = fPtrSolidB->DistanceToOut( */
/*                         fPtrSolidB->transformation()->Transform(p+dist*v),v); */
/* 	    // std::cout << "VecdistTmp2 " << disTmp << "\n"; */
/*             dist += disTmp; */
/*             //dist += (disTmp>=0.)? disTmp : 0.; */
/*          } */
/*         //if(count1 > 100){ */
/*             //std::cerr << "LOOP1 INFINITY\n"; break; } */
/*        } */
/*        while( fPtrSolidA->Inside(p+dist*v) != EInside::kOutside && */
/*                      disTmp > kHalfTolerance ) ; */
/*       // NOTE was kCarTolerance; just taking kHalfHolerance */
/*     } */
/*     else // if( positionB != kOutside ) */
/*     { */
/*       do */
/*       { */
/*         count2++; */
/* 	disTmp = fPtrSolidB->DistanceToOut( */
/*                    fPtrSolidB->transformation()->Transform(p+dist*v),v); */
/* 	//  std::cout << "VecdistTmp3 " << disTmp << "\n"; */
/*         //dist += disTmp ; */
/*         dist += (disTmp>=0. && disTmp<kInfinity)? disTmp : 0.; */
/*         if(fPtrSolidA->Inside(p+dist*v) != EInside::kOutside) */
/*            { */
/*              disTmp = fPtrSolidA->DistanceToOut(p+dist*v,v); */
/*              //std::cerr << "distTmp4 " << disTmp; */
/* 	     //   std::cout << "VecdistTmp4 " << disTmp << "\n"; */
/*              dist += disTmp ; */
/*              //dist += (disTmp>=0.)? disTmp : 0.; */
/*            } */

/*         if(count2 > 100){ */
/*              std::cerr << "LOOP2 INFINITY\n"; break; } */
/* 	//  std::cout << fPtrSolidB->Contains(p+dist*v) << "\n"; */
/* 	// std::cout << fPtrSolidB->Inside(p+dist*v) << "\n"; */
/*       } */
/*       while( (fPtrSolidB->Inside(p+dist*v) != EInside::kOutside) */
/*                && (disTmp > kHalfTolerance) ); */
/*     } */
/*     //  std::cerr << "--VecGeom return " << dist << "\n"; */
/*     distance = dist; */
/*     return; */
/* } */


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;
  Float_t distA = fPtrSolidA->SafetyToIn(p);
  Float_t distB = fPtrSolidB->SafetyToIn(p);
  safety = Min(distA, distB);
  MaskedAssign( safety < 0, 0., &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety) 
	{

    VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
    VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

    typedef typename Backend::bool_v Bool_t;

    Bool_t containedA = fPtrSolidA->Contains(p) ;
    Bool_t containedB = fPtrSolidB->Contains(p) ;

    if( containedA && containedB ) /* in both */
    {
      safety= Max(fPtrSolidA->SafetyToOut(p),
                  fPtrSolidB->SafetyToOut(
                          fPtrSolidB->GetTransformation()->Transform(p)
                          ) ) ; // is max correct ??
    }
    else
    {
     if( containedB ) /* only contained in B */
      {
        safety = fPtrSolidB->SafetyToOut(
                    fPtrSolidB->GetTransformation()->Transform(p));
      }
      else
      {
        safety = fPtrSolidA->SafetyToOut(p) ;
      }
    }
	return;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::NormalKernel(
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
