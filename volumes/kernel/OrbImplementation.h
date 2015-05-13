
/// @file OrbImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedOrb.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(OrbImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE { 

class PlacedOrb;
class UnplacedOrb;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct OrbImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;


    
using PlacedShape_t = PlacedOrb;
using UnplacedShape_t = UnplacedOrb;

VECGEOM_CUDA_HEADER_BOTH
static void PrintType() {
   printf("SpecializedOrb<%i, %i>", transCodeT, rotCodeT);
}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);
    
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedOrb const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedOrb const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside);

template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void GenericKernelForContainsAndInside(
    UnplacedOrb const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) ;

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedOrb const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedOrb const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

 template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);
 

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void ContainsKernel(
    UnplacedOrb const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside);
  

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void InsideKernel(
    UnplacedOrb const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) ;

 
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);


template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);


template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(UnplacedOrb const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);
  


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);
  
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Normal(
       UnplacedOrb const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedOrb const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );
  
  
};

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::Normal(
       UnplacedOrb const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){

    NormalKernel<Backend>(unplaced, point, normal, valid);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::NormalKernel(
       UnplacedOrb const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){

    typedef typename Backend::precision_v Double_t;
    // typedef typename Backend::bool_v      Bool_t;

    Vector3D<Double_t> localPoint;
    localPoint = point;

    Double_t rad2=localPoint.Mag2();
    Double_t rad=Sqrt(rad2);
    normal = Vector3D<Double_t>(localPoint.x()/rad , localPoint.y()/rad ,localPoint.z()/rad );
    
    
    Double_t tolRMaxP = unplaced.GetfRTolO();
    Double_t tolRMaxM = unplaced.GetfRTolI();

    // Check radial surface
    valid = ((rad2 <= tolRMaxP * tolRMaxP) && (rad2 >= tolRMaxM * tolRMaxM)); // means we are on surface
   
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::UnplacedContains(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside){

      ContainsKernel<Backend>(unplaced, point, inside);
}    

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::Contains(UnplacedOrb const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside){

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::Inside(UnplacedOrb const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside){

    InsideKernel<Backend>(unplaced,
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::DistanceToIn(UnplacedOrb const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    DistanceToInKernel<Backend>(
            unplaced,
            transformation.Transform<transCodeT, rotCodeT>(point),
            transformation.TransformDirection<rotCodeT>(direction),
            stepMax,
            distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::DistanceToOut(UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    DistanceToOutKernel<Backend>(
    unplaced,
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
void OrbImplementation<transCodeT, rotCodeT>::SafetyToIn(UnplacedOrb const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){
    
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
void OrbImplementation<transCodeT, rotCodeT>::SafetyToOut(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
    SafetyToOutKernel<Backend>(
    unplaced,
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::ContainsKernel(UnplacedOrb const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  typedef typename Backend::bool_v Bool_t;
  Bool_t unused;
  Bool_t outside;
  GenericKernelForContainsAndInside<Backend, false>(unplaced,
    localPoint, unused, outside);
  inside=!outside;
}  

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
UnplacedOrb const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

    typedef typename Backend::precision_v Double_t;
    // typedef typename Backend::bool_v      Bool_t;

    Double_t rad2 = localPoint.Mag2();
    Double_t tolRMax = unplaced.GetfRTolO();
    Double_t tolRMin = unplaced.GetfRTolI();
    Double_t tolRMax2 = tolRMax * tolRMax;
    Double_t tolRMin2 = tolRMin * tolRMin;
    
    completelyoutside = ( rad2 > tolRMax2);
    if (ForInside)
    {
      completelyinside = ( rad2 < tolRMin2);
    }
    
    
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::InsideKernel(UnplacedOrb const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  
  typedef typename Backend::bool_v      Bool_t;
  Bool_t completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Backend,true>(
      unplaced, point, completelyinside, completelyoutside);
  inside=EInside::kSurface;
  MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  MaskedAssign(completelyinside, EInside::kInside, &inside);
}
 
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){
    
    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Double_t> localPoint;
    localPoint = point;

    Vector3D<Double_t> localDir;
    localDir = direction;

    //General Precalcs
    Double_t rad2 = localPoint.Mag2();
    Double_t rad = Sqrt(rad2);
    Double_t pDotV3d = localPoint.Dot(localDir);

    Double_t radius2 = unplaced.GetRadius() * unplaced.GetRadius();
    Double_t c = rad2 - radius2;
    Double_t d2 = pDotV3d * pDotV3d - c;

    Double_t pos_dot_dir_x = localPoint.x()*localDir.x();
    Double_t pos_dot_dir_y = localPoint.y()*localDir.y();
    Double_t pos_dot_dir_z = localPoint.z()*localDir.z();

    Bool_t done(false);
    distance = kInfinity;
    Double_t zero=Backend::kZero;

    //Is the point Inside
    Bool_t isInside = ((rad < unplaced.GetfRTolI()));
    done |= isInside;
    MaskedAssign( isInside, kInfinity, &distance );
    if( IsFull(done) )return;

    //On the Surface and Moving In
    Bool_t isInsideOuterToleranceAndMovingIn=((rad >= unplaced.GetfRTolI()) && (rad <= unplaced.GetfRTolO()) && (pDotV3d < 0));
    done |= isInsideOuterToleranceAndMovingIn;
    MaskedAssign(isInsideOuterToleranceAndMovingIn,zero,&distance);
    if( IsFull(done) )return;

    //On the Surface and Moving Out
    Bool_t isInsideOuterToleranceAndMovingOut=((rad >= unplaced.GetfRTolI()) && (rad <= unplaced.GetfRTolO()) && (pDotV3d >= 0));
    done |= isInsideOuterToleranceAndMovingOut;
    MaskedAssign(isInsideOuterToleranceAndMovingIn,kInfinity,&distance);
    if( IsFull(done) )return;
    
    //Outside the Surface and Moving In
    Bool_t isOutsideOuterToleranceAndMovingIn=( (rad > unplaced.GetfRTolO()) && (pDotV3d < 0));
    done |= isOutsideOuterToleranceAndMovingIn;
    MaskedAssign(isOutsideOuterToleranceAndMovingIn,(-pDotV3d - Sqrt(d2)),&distance);
    if( IsFull(done) )return;
    
    //Outside the Surface and Moving Out
    Bool_t isOutsideOuterToleranceAndMovingOut=( (rad > unplaced.GetfRTolO()) && (pDotV3d >= 0));
    done |= isOutsideOuterToleranceAndMovingOut;
    MaskedAssign(isOutsideOuterToleranceAndMovingOut,kInfinity,&distance);
    if( IsFull(done) )return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

   typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

    distance = kInfinity;  
    Double_t zero=Backend::kZero;
    distance = zero;
    
    Vector3D<Double_t> localPoint;
    localPoint = point;
    
    Vector3D<Double_t> localDir;
    localDir =  direction;
    
    //General Precalcs
    Double_t rad2    = localPoint.Mag2();
    Double_t rad = Sqrt(rad2);
    Double_t pDotV3d = localPoint.Dot(localDir);

    Double_t radius2 = unplaced.GetRadius() * unplaced.GetRadius();
    Double_t c = rad2 - radius2;
    Double_t d2 = pDotV3d * pDotV3d - c;
  
    Bool_t done(false);
    distance = kInfinity;
    distance = zero;
    
    //checking if the point is outside
    Double_t tolRMax = unplaced.GetfRTolO();
    Double_t tolRMax2 = tolRMax * tolRMax;
    Bool_t isOutside = ( rad2 > tolRMax2);
    done|= isOutside;
    if( IsFull(done) )return;

    Bool_t isInsideAndWithinOuterTolerance = ((rad <= tolRMax) && (c < (kTolerance * unplaced.GetRadius())));
    Bool_t isInsideAndOnTolerantSurface = ((c > (-2*kTolerance*unplaced.GetRadius())) && ( (pDotV3d >= 0) || (d2 < 0) ));

    Bool_t onSurface=(isInsideAndWithinOuterTolerance && isInsideAndOnTolerantSurface );
    MaskedAssign(onSurface , zero, &distance);
    done|=onSurface;
    if( IsFull(done) )return;

    Bool_t notOnSurface=(isInsideAndWithinOuterTolerance && !isInsideAndOnTolerantSurface );
    MaskedAssign(notOnSurface , (-pDotV3d + Sqrt(d2)), &distance);
    done|=notOnSurface;
    if( IsFull(done) )return;
    
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::SafetyToInKernel(UnplacedOrb const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){

    typedef typename Backend::precision_v Double_t;
    // typedef typename Backend::bool_v      Bool_t;

    Double_t safe=Backend::kZero;
    Double_t zero=Backend::kZero; 

    Vector3D<Double_t> localPoint;
    //localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    localPoint=point;

    //General Precalcs
    Double_t rad2    = localPoint.Mag2();
    Double_t rad = Sqrt(rad2);
    safe = rad - unplaced.GetRadius();
    safety = safe;
    MaskedAssign( (safe < zero) , zero, &safety);
    
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
    
    typedef typename Backend::precision_v Double_t;
    // typedef typename Backend::bool_v      Bool_t;

    Double_t safe=Backend::kZero;
    Double_t zero=Backend::kZero; 

    Vector3D<Double_t> localPoint;
    localPoint = point;
    
    //General Precalcs
    Double_t rad2    = localPoint.Mag2();
    Double_t rad = Sqrt(rad2);
    safe = unplaced.GetRadius() - rad;
    safety = safe;
    MaskedAssign( (safe < zero) , zero, &safety);
}






} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
