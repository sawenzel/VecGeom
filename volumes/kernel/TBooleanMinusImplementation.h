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
#include "volumes/UnplacedBox.h"
#include "volumes/kernel/GenericKernels.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct TBooleanMinusImplementation {

  //
  template<typename Unplaced_t, typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      Unplaced_t const &unplaced,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      Unplaced_t const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      Unplaced_t const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      Unplaced_t const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      Unplaced_t const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(Unplaced_t const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(Unplaced_t const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      Unplaced_t const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      Unplaced_t const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      Unplaced_t const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      Unplaced_t const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      Unplaced_t const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      Unplaced_t const & unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Unplaced_t, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       Unplaced_t const & unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );

}; // End struct TBooleanMinusImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::UnplacedContains(
    Unplaced_t const & unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Unplaced_t, Backend>( unplaced , localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::Contains(
    Unplaced_t const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Unplaced_t, Backend>(unplaced, localPoint, inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::Inside(
    Unplaced_t const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  InsideKernel<Unplaced_t, Backend>(unplaced,
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToIn(
    Unplaced_t const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToInKernel<Unplaced_t, Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    transformation.TransformDirection<rotCodeT>(direction),
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToOut(
    Unplaced_t const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Unplaced_t, Backend>(
    unplaced,
    point,
    direction,
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TBooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToIn(
    Unplaced_t const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToInKernel<Unplaced_t, Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TBooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToOut(
    Unplaced_t const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToOutKernel<Unplaced_t, Backend>(
    unplaced,
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::ContainsKernel(
    Unplaced_t const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  // now just use the Contains functionality
  // of Unplaced and its left and right components
  // Find if a subtraction of two shapes contains a given point
   inside = unplaced.fLeftVolume->Contains(localPoint);
   if ( IsEmpty(inside) ) return Backend::kFalse;
   inside &= ! unplaced.fRightVolume->Contains(localPoint);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::InsideKernel(
    Unplaced_t const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  // now use the Inside functionality of left and right components

  // going to be a bit more complicated due to Surface states
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    Unplaced_t const &unplaced,
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

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    Unplaced_t const & unplaced,
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
    distance = unplaced.fLeftVolume->Unplaced_t::LeftType::DistanceToOut( point, direction, stepMax );
    Float_t dinright = unplaced.fRightVolume->Unplaced_t::RightType::DistanceToIn( point, direction, stepMax );

    distance = Min( distance, dinright );
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    Unplaced_t const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  // this is a very crude algorithm which as very many "virtual" surfaces
  // which could return a very small safety
  Bool_t  inleft   = unplaced.fLeftVolume->Contains(point);
  Float_t safleft  = unplaced.fLeftVolume->SafetyToIn(point);
  Bool_t  inright  = unplaced.fRightVolume->Contains(point);
  Float_t safright = unplaced.fRightVolume->SafetyToOut(point);
  Bool_t done(Backend::kFalse);
  MaskedAssign( inleft && inright, safright, &safety );
  done |= inleft && inright;
  MaskedAssign( !done && inleft, Min(safleft, safright), &safety);
  done |= inleft;
  MaskedAssign( !done && inright, Max(safleft, safright), &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    Unplaced_t const & unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::bool_v Bool_t;
   typedef typename Backend::precision_v Float_t;

   Float_t safleft  = unplaced.fLeftVolume->SafetyToOut(point);
   Float_t safright = unplaced.fRightVolume->SafetyToIn(point);
   safety = Min(safleft, safright);
    // might return a negative number --> navigator will handle this
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Unplaced_t, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void TBooleanMinusImplementation<transCodeT, rotCodeT>::NormalKernel(
     Unplaced_t const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {
    // TBDONE
}

} // End global namespace



#endif /* TBOOLEANMINUSIMPLEMENTATION_H_ */
