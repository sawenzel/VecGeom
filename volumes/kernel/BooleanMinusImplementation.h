/*
 * BooleanMinusImplementation.h
 */

#ifndef BOOLEANMINUSIMPLEMENTATION_H_
#define BOOLEANMINUSIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBooleanMinusVolume.h"

namespace VECGEOM_NAMESPACE {



/**
 * an ordinary (non-templated) implementation of a BooleanMinus solid
 * using the virtual function interface of its constituents
 * Note that the Backend here is usually not kVc
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanMinusImplementation {

  //
  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedBooleanMinusVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedBooleanMinusVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedBooleanMinusVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedBooleanMinusVolume const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedBooleanMinusVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedBooleanMinusVolume const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedBooleanMinusVolume const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      UnplacedBooleanMinusVolume const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      UnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      UnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      UnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      UnplacedBooleanMinusVolume const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedBooleanMinusVolume const & unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );

}; // End struct BooleanMinusImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>( unplaced , localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::Contains(
        UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::ContainsKernel(
    UnplacedBooleanMinusVolume const &unplaced,
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
void BooleanMinusImplementation<transCodeT, rotCodeT>::InsideKernel(
    UnplacedBooleanMinusVolume const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  // now use the Inside functionality of left and right components

  // going to be a bit more complicated due to Surface states
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedBooleanMinusVolume const &unplaced,
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
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    distance = unplaced.fLeftVolume->DistanceToOut( point, direction, stepMax );
    Float_t dinright = unplaced.fRightVolume->DistanceToIn( point, direction, stepMax );
    distance = Min( distance, dinright );
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedBooleanMinusVolume const & unplaced,
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

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedBooleanMinusVolume const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::bool_v Bool_t;
   typedef typename Backend::precision_v Float_t;

   safety = unplaced.fLeftVolume->SafetyToOut( point );
   Float_t safetyright = unplaced.fRightVolume->SafetyToIn( point);
   safety = Min( safety, safetyright );
   return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BooleanMinusImplementation<transCodeT, rotCodeT>::NormalKernel(
     UnplacedBooleanMinusVolume const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {
    // TBDONE
}

} // End global namespace



#endif /* TBOOLEANMINUSIMPLEMENTATION_H_ */
