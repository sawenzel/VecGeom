/*
 * TBooleanMinusImplementation.h
 *
 *  Created on: Aug 13, 2014
 *      Author: swenzel
 */

#ifndef TBOOLEANMINUSIMPLEMENTATION_H_
#define TBOOLEANMINUSIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/TUnplacedBooleanMinusVolume.h"

namespace VECGEOM_NAMESPACE {

template <typename LeftPlacedType_t, typename RightPlacedType_t, TranslationCode transCodeT, RotationCode rotCodeT>
struct TBooleanMinusImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  //
  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      TUnplacedBooleanMinusVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      TUnplacedBooleanMinusVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      TUnplacedBooleanMinusVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      TUnplacedBooleanMinusVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      TUnplacedBooleanMinusVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(TUnplacedBooleanMinusVolume const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(TUnplacedBooleanMinusVolume const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      TUnplacedBooleanMinusVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      TUnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      TUnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      TUnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      TUnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      TUnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       TUnplacedBooleanMinusVolume const & unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );

}; // End struct TBooleanMinusImplementation

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::UnplacedContains(
    TUnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>( unplaced , localPoint, inside);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::Contains(
        TUnplacedBooleanMinusVolume const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);

}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::Inside(
    TUnplacedBooleanMinusVolume const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  InsideKernel<Backend>(unplaced,
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::DistanceToIn(
    TUnplacedBooleanMinusVolume const &unplaced,
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

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::DistanceToOut(
    TUnplacedBooleanMinusVolume const &unplaced,
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

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::SafetyToIn(
    TUnplacedBooleanMinusVolume const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToInKernel<Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    safety
  );
}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::SafetyToOut(
    TUnplacedBooleanMinusVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToOutKernel<Backend>(
    unplaced,
    point,
    safety
  );
}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::ContainsKernel(
    TUnplacedBooleanMinusVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  // now just use the Contains functionality
  // of Unplaced and its left and right components
  // Find if a subtraction of two shapes contains a given point

   // have to figure this out
   Vector3D<typename Backend::precision_v> tmp;
   LeftPlacedType_t::Implementation::template Contains<Backend>(
           *((LeftPlacedType_t *) unplaced.fLeftVolume)->GetUnplacedVolume(),
           *unplaced.fLeftVolume->transformation(),
           localPoint, tmp, inside);

//   TUnplacedBooleanMinusVolume const &unplaced,
//        Transformation3D const &transformation,
//        Vector3D<typename Backend::precision_v> const &point,
//        Vector3D<typename Backend::precision_v> &localPoint,
//        typename Backend::bool_v &inside
//

   if ( IsEmpty(inside) ) return;

   typename Backend::bool_v rightInside;
   RightPlacedType_t::Implementation::template Contains<Backend>(
           *((RightPlacedType_t *) unplaced.fRightVolume)->GetUnplacedVolume(),
           *unplaced.fRightVolume->transformation(),
           localPoint, tmp, rightInside);

   inside &= ! rightInside;
}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::InsideKernel(
    TUnplacedBooleanMinusVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  // now use the Inside functionality of left and right components

  // going to be a bit more complicated due to Surface states
}


template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::DistanceToInKernel(
    TUnplacedBooleanMinusVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v      Bool_t;

  // TOBEDONE: ASK Andrei about the while loop
  // Compute distance from a given point outside to the shape.
//  Int_t i;
//  Float_t d1, d2, snxt=0.;
//  fRightMat->MasterToLocal(point, &local[0]);
//  fLeftMat->MasterToLocalVect(dir, &ldir[0]);
//  fRightMat->MasterToLocalVect(dir, &rdir[0]);
//
//  // check if inside '-'
//  Bool_t insideRight = unplaced.fRightVolume->Contains(point);
//  // epsilon is used to push across boundaries
//  Precision epsil(0.);
//
//  // we should never subtract a volume such that B - A > 0
//
//  // what does this while loop do?
//  if ( ! IsEmpty( insideRight ) ) {
//    // propagate to outside of '- / RightShape'
//    d1 = unplaced.fRightVolume->DistanceToOut( point, direction, stepMax);
//    snxt += d1+epsil;
//    hitpoint += (d1+1E-8)*direction;
//
//    epsil = 1.E-8;
//    // now master outside 'B'; check if inside 'A'
//    Bool_t insideLeft =
//    if (unplaced.fLeftVolume->Contains(&local[0])) return snxt;
//  }
//
//  // if outside of both we do a max operation
//  // master outside '-' and outside '+' ;  find distances to both
//        node->SetSelected(1);
//        fLeftMat->MasterToLocal(&master[0], &local[0]);
//        d2 = fLeft->DistFromOutside(&local[0], &ldir[0], iact, step, safe);
//        if (d2>1E20) return TGeoShape::Big();
//
//        fRightMat->MasterToLocal(&master[0], &local[0]);
//        d1 = fRight->DistFromOutside(&local[0], &rdir[0], iact, step, safe);
//        if (d2<d1-TGeoShape::Tolerance()) {
//           snxt += d2+epsil;
//           return snxt;
//        }
//        // propagate to '-'
//        snxt += d1+epsil;
//        for (i=0; i<3; i++) master[i] += (d1+1E-8)*dir[i];
//        epsil = 1.E-8;
//        // now inside '-' and not inside '+'
//        fRightMat->MasterToLocal(&master[0], &local[0]);
//        inside = kTRUE;
//     }

}
#include <iostream>
template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::DistanceToOutKernel(
    TUnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // how can we force to inline this ?
    // Left is an unplaced shape; but it could be a complicated one

    // what happens if LeftType is VPlacedVolume itself?
    // For the SOA case: is it better to push SOA down or do ONE loop around?
   // distance = unplaced.fLeftVolume->Unplaced_t::LeftType::DistanceToOut( point, direction, stepMax );
   // Float_t dinright = unplaced.fRightVolume->Unplaced_t::RightType::DistanceToIn( point, direction, stepMax );

    // we need a template specialization for this in case we have LeftType or RightType equals to
    // VPlacedVolume

    LeftPlacedType_t::Implementation::template DistanceToOut<Backend>(
                    *((LeftPlacedType_t *) unplaced.fLeftVolume)->GetUnplacedVolume(),
                    point, direction, stepMax, distance );
    Float_t dinright(kInfinity);
    RightPlacedType_t::Implementation::template DistanceToIn<Backend>(
                   *((RightPlacedType_t *) unplaced.fRightVolume)->GetUnplacedVolume(),
                    *unplaced.fRightVolume->transformation(),
                    point, direction, stepMax, dinright );
    distance = Min( distance, dinright );
    return;
}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::SafetyToInKernel(
    TUnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

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

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t, transCodeT, rotCodeT>::SafetyToOutKernel(
    TUnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::bool_v Bool_t;
   typedef typename Backend::precision_v Float_t;

   LeftPlacedType_t::Implementation::template SafetyToOut<Backend>(
                      *((LeftPlacedType_t *) unplaced.fLeftVolume)->GetUnplacedVolume(),
                      point, safety );
   Float_t safetyright(kInfinity);
   RightPlacedType_t::Implementation::template SafetyToIn<Backend>(
                     *((RightPlacedType_t *) unplaced.fRightVolume)->GetUnplacedVolume(),
                      *unplaced.fRightVolume->transformation(),
                      point, safetyright );
   safety = Min( safety, safetyright );
   return;
}

template <typename LeftPlacedType_t, typename RightPlacedType_t,TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<LeftPlacedType_t,RightPlacedType_t,transCodeT, rotCodeT>::NormalKernel(
     TUnplacedBooleanMinusVolume const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {
    // TBDONE
}

} // End global namespace



#endif /* TBOOLEANMINUSIMPLEMENTATION_H_ */
