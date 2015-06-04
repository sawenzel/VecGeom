/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANIMPLEMENTATION_H_
#define BOOLEANIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBooleanVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(BooleanImplementation, BooleanOperation, Arg1, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBooleanVolume;
class UnplacedBooleanVolume;

template <BooleanOperation boolOp, TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanImplementation {
   static const int transC = transCodeT;
   static const int rotC   = rotCodeT;
   using PlacedShape_t = PlacedBooleanVolume;
   using UnplacedShape_t = UnplacedBooleanVolume;

    // empty since functionality will be implemented in
// partially template specialized structs
};


/**
 * an ordinary (non-templated) implementation of a Boolean solid
 * using the virtual function interface of its constituents
 * Note that the Backend here is usually not kVc
 *
 * TEMPLATE SPECIALIZATION FOR UNION
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanImplementation<kSubtraction, transCodeT, rotCodeT> {

   static const int transC = transCodeT;
   static const int rotC   = rotCodeT;

   using PlacedShape_t = PlacedBooleanVolume;
   using UnplacedShape_t = UnplacedBooleanVolume;

   VECGEOM_CUDA_HEADER_BOTH
   static void PrintType() {
     printf("SpecializedBooleanVolume<%i, %i, %i>", kSubtraction, transCodeT, rotCodeT);
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>( unplaced , localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::Contains(
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::Inside(
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::DistanceToIn(
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::DistanceToOut(
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::SafetyToIn(
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::SafetyToOut(
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
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::ContainsKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  // now just use the Contains functionality
  // of Unplaced and its left and right components
  // Find if a subtraction of two shapes contains a given point

   // have to figure this out
   Vector3D<typename Backend::precision_v> tmp;

   inside = unplaced.fLeftVolume->Contains(localPoint);
   if ( IsEmpty(inside) ) return;

   typename Backend::bool_v rightInside = unplaced.fRightVolume->Contains(localPoint);
   inside &= ! rightInside;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::InsideKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::inside_v &inside) {

  // now use the Inside functionality of left and right components
  // algorithm taken from Geant4 implementation
  VPlacedVolume const*const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const*const fPtrSolidB = unplaced.fRightVolume;

  typename Backend::inside_v positionA = fPtrSolidA->Inside(p);
  if ( positionA == EInside::kOutside ){
        inside = EInside::kOutside;
        return;
  }

  typename Backend::inside_v positionB = fPtrSolidB->Inside(p);

  if(positionA == EInside::kInside && positionB == EInside::kOutside)
  {
     inside = EInside::kInside;
     return;
  }
  else {
    if(( positionA == EInside::kInside  && positionB == EInside::kSurface) ||
       ( positionB == EInside::kOutside && positionA == EInside::kSurface)
/*
         || ( positionA == EInside::kSurface && positionB == EInside::kSurface &&
           ( fPtrSolidA->Normal(p) -
             fPtrSolidB->Normal(p) ).mag2() >
           1000.0*G4GeometryTolerance::GetInstance()->GetRadialTolerance() ) )
*/ )
      {
        inside = EInside::kSurface;
        return;
      }
      else
      {
        inside = EInside::kOutside;
        return;
      }
    }
  // going to be a bit more complicated due to Surface states
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedBooleanVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v      Bool_t;

  // for the moment we take implementation from Geant4
//  Float_t dist = 0.0,disTmp = 0.0 ;
//
//  VPlacedVolume const* const fPtrSolidB = unplaced.fRightVolume;
//  VPlacedVolume const* const fPtrSolidA = unplaced.fLeftVolume;
//
//  if ( fPtrSolidB->Contains(p) )   // start: out of B if inside B
//     {
//       dist = fPtrSolidB->PlacedDistanceToOut( p, v ); // ,calcNorm,validNorm,n) ;
//
//       if( ! fPtrSolidA->Contains(p+dist*v) ){
//         int count1=0;
//         Bool_t contains;
//         do {
//          disTmp = fPtrSolidA->DistanceToIn(p+dist*v,v) ;
//
//          if(disTmp == kInfinity) {
//              distance=kInfinity;
//              return;
//          }
//          dist += disTmp ;
//
//          // Contains is the one from this boolean solid
//          UnplacedContains<Backend>( unplaced, p+dist*v, contains );
//          if( IsEmpty(contains) ) /* if not contained */
//          {
//            disTmp = fPtrSolidB->PlacedDistanceToOut(p+dist*v,v);
//            dist += disTmp ;
//            count1++;
//            if( count1 > 1000 )  // Infinite loop detected
//            {
//               // EMIT WARNING
//            }
//          }
//          UnplacedContains<Backend>( unplaced, p+dist*v, contains );
//         }
//        while( IsEmpty(contains) );
//       }
//       // no else part
//    }
//  else // p outside B and of A;
//    {
//       dist = fPtrSolidA->DistanceToIn(p,v) ;
//
//       if( dist == kInfinity ) // past A, hence past A\B
//       {
//         distance=kInfinity;
//           return;
//       }
//       else
//       {
//         int count2=0;
//         Bool_t contains;
//         UnplacedContains<Backend>(unplaced, p+dist*v, contains);
//         while( IsEmpty(contains) )  // pushing loop
//         {
//           disTmp = fPtrSolidB->PlacedDistanceToOut(p+dist*v,v);
//           dist += disTmp ;
//           dist += kTolerance;
//
//           UnplacedContains<Backend>(unplaced, p+dist*v, contains);
//           if( IsEmpty(contains) )
//           {
//             disTmp = fPtrSolidA->DistanceToIn(p+dist*v,v) ;
//
//             if(disTmp == kInfinity) // past A, hence past A\B
//             {
//               distance=kInfinity;
//                 return;
//             }
//             dist += disTmp ;
//             count2++;
//             if( count2 > 1000 )  // Infinite loop detected
//             {
//               // EMIT WARNING
//             }
//           }
//         }
//       }
//     }
//    distance=dist;
//   return;



  // TOBEDONE: ASK Andrei about the while loop
  // Compute distance from a given point outside to the shape.
//  Int_t i;
    Float_t d1, d2, snxt=0.;
    Vector3D<Precision> hitpoint=p;
    //  fRightMat->MasterToLocal(point, &local[0]);
//  fLeftMat->MasterToLocalVect(dir, &ldir[0]);
//  fRightMat->MasterToLocalVect(dir, &rdir[0]);
//
  // check if inside '-'
    Bool_t insideRight = unplaced.fRightVolume->Contains(p);
//  // epsilon is used to push across boundaries
    Precision epsil(0.);
//
//  // we should never subtract a volume such that B - A > 0
//
//  // what does this while loop do?
   while(1){
    if ( insideRight  ) {
//    // propagate to outside of '- / RightShape'
      d1 = unplaced.fRightVolume->PlacedDistanceToOut( hitpoint, v, stepMax);
      snxt += (d1 > 0. && d1 < kInfinity)? d1+epsil : 0.;
      hitpoint += (d1 > 0. && d1 < kInfinity)? (d1+1E-8)*v : 1E-8*v;

//
      epsil = 1.E-8;
// now master outside 'B'; check if inside 'A'
//    Bool_t insideLeft =
    if (unplaced.fLeftVolume->Contains( hitpoint )){
        distance = snxt;
	//	std::cerr << "hitting  " << distance << "\n";
        return;
    }
  }

  // if outside of both we do a max operation
  // master outside '-' and outside '+' ;  find distances to both
//        fLeftMat->MasterToLocal(&master[0], &local[0]);
   d2 = unplaced.fLeftVolume->DistanceToIn( hitpoint, v, stepMax );
   if (d2 == kInfinity){
       distance = kInfinity;
       // std::cerr << "missing A " << d2 << "\n";
       return ;
   }

   d1 = unplaced.fRightVolume->DistanceToIn( hitpoint, v, stepMax );
   if (d2 < d1 - kTolerance ) {
          snxt += d2+epsil;
	  // std::cerr << "returning  " << snxt << "\n";
          distance = snxt;
          return;
   }

   //        // propagate to '-'
   snxt += (d1>0. && d1<kInfinity)?  d1+epsil : 0.;
   hitpoint += (d1>0. && d1<kInfinity)? (d1 + epsil)*v : epsil*v;
   insideRight = true;
  } // end while
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;

    distance = unplaced.fLeftVolume->DistanceToOut( point, direction, stepMax );
    Float_t dinright = unplaced.fRightVolume->DistanceToIn( point, direction, stepMax );
    distance = Min( distance, dinright );
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety) {

  VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

  // very approximate
  if( ( fPtrSolidA->Contains(p) ) &&   // case 1
        ( fPtrSolidB->Contains(p) ) )
    {
        safety = fPtrSolidB->SafetyToOut(
					 fPtrSolidB->GetTransformation()->Transform(p));
    }
    else
    {
        // po
        safety = fPtrSolidA->SafetyToIn(p);
    }
    return;

  // this is a very crude algorithm which as very many "virtual" surfaces
  // which could return a very small safety
//  Bool_t  inleft   = unplaced.fLeftVolume->Contains(point);
//  Float_t safleft  = unplaced.fLeftVolume->SafetyToIn(point);
//  Bool_t  inright  = unplaced.fRightVolume->Contains(point);
//  Float_t safright = unplaced.fRightVolume->SafetyToOut(point);
//  Bool_t done(Backend::kFalse);
//  MaskedAssign( inleft && inright, safright, &safety );
//  done |= inleft && inright;
//  MaskedAssign( !done && inleft, Min(safleft, safright), &safety);
//  done |= inleft;
//  MaskedAssign( !done && inright, Max(safleft, safright), &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedBooleanVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::precision_v Float_t;

   safety = unplaced.fLeftVolume->SafetyToOut( point );
   Float_t safetyright = unplaced.fRightVolume->SafetyToIn( point);
   safety = Min( safety, safetyright );
   return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanImplementation<kSubtraction, transCodeT, rotCodeT>::NormalKernel(
     UnplacedBooleanVolume const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {
    // TBDONE
}

} // End impl namespace

} // End global namespace

// include stuff for boolean union
#include "BooleanUnionImplementation.h"

// include stuff for boolean intersection
#include "BooleanIntersectionImplementation.h"

#endif /* BOOLEANIMPLEMENTATION_H_ */
