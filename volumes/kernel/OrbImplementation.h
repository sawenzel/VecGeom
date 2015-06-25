
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

    typedef typename Backend::precision_v Float_t;

    Vector3D<Float_t> localPoint;
    localPoint = point;

    Float_t rad2=localPoint.Mag2();
    Float_t rad=Sqrt(rad2);
    normal = Vector3D<Float_t>(localPoint.x()/rad , localPoint.y()/rad ,localPoint.z()/rad );
    
    Float_t tolRMaxP = unplaced.GetfRTolO();
    Float_t tolRMaxM = unplaced.GetfRTolI();

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

    
    typedef typename Backend::precision_v Float_t;
    //typedef typename Backend::bool_v      Bool_t;	
        
    Precision fR = unplaced.GetRadius();
    Float_t rad2 = localPoint.Mag2();
    
    Float_t tolR = fR - ( kTolerance );
    if(ForInside)
    completelyinside = (rad2 <= tolR *tolR) ;
    
    tolR = (fR + ( kTolerance)); 
    completelyoutside = (rad2 >= tolR *tolR);
    //if( IsFull(completelyoutside) )return;

    //Radial Check for GenericKernel Over
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
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;
    //typedef typename Backend::inside_v    Inside_t;

    Vector3D<Float_t> localPoint = point;
    Vector3D<Float_t> localDir = direction;

    distance = kInfinity;
    Bool_t done(false);

    Float_t fR(unplaced.GetRadius()); 
	// General Precalcs
    Float_t rad2 = localPoint.Mag2();
    Float_t pDotV3d = localPoint.Dot(localDir);

    Float_t  c(0.), d2(0.);
    c = rad2 - fR * fR;
    //MaskedAssign((tr),(pDotV3d * pDotV3d - c),&d2);
    d2 = (pDotV3d * pDotV3d - c);

    done |= (d2 < 0. || ((localPoint.Mag() > fR) && (pDotV3d > 0.)));
    if(IsFull(done)) return; //Returning in case of no intersection with outer shell

    Bool_t test1 = !done && (pDotV3d < 0.0);
    Bool_t test2 = (Sqrt(rad2) >= (fR - kTolerance) ) && (Sqrt(rad2) <= (fR + kTolerance));
    Bool_t test3 = (Sqrt(rad2) > (fR + kTolerance) ) && (d2 >= 0.0);

    MaskedAssign(test1 && test2, 0.0, &distance);
    MaskedAssign(test1 && test3,-1.0 * pDotV3d - Sqrt(d2), &distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      /*Vector3D<typename Backend::precision_v> const &n,  
      typename Backend::bool_v validNorm,  */
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Float_t> localPoint = point;
    Vector3D<Float_t> localDir=direction;

    distance = kInfinity;
    Float_t  pDotV2d, pDotV3d;

    Bool_t done(false);
    Float_t fR(unplaced.GetRadius()); 

    // Intersection point
    Vector3D<Float_t> intSecPt;
    Float_t  c(0.), d2(0.);

    pDotV2d = localPoint.x() * localDir.x() + localPoint.y() * localDir.y();
    pDotV3d = pDotV2d + localPoint.z() * localDir.z(); //localPoint.Dot(localDir);

    Float_t rad2 = localPoint.Mag2();
    c = rad2 - fR * fR;

   //New Code
   Float_t sd1(0.);

   Bool_t cond1 = (Sqrt(rad2) <= (fR + 0.5*kTolerance)) ;
   Bool_t cond = (Sqrt(rad2) <= (fR + kTolerance)) && (Sqrt(rad2) >= (fR - kTolerance)) && pDotV3d >=0 && cond1;
   done |= cond;
   MaskedAssign(cond ,0.,&sd1);

   MaskedAssign(cond1, (pDotV3d * pDotV3d - c), &d2);
   MaskedAssign((!done && cond1 && (d2 >= 0.0)), (-1.*pDotV3d + Sqrt(d2)), &sd1);

   MaskedAssign((sd1 < 0.),kInfinity, &sd1);
   distance=sd1;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::SafetyToInKernel(UnplacedOrb const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){

    typedef typename Backend::precision_v Float_t;
    
    Float_t safe(0.);
    Vector3D<Float_t> localPoint;
    localPoint=point;

    //General Precalcs
    Float_t rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2);
    safe = rad - unplaced.GetRadius();
    safety = safe;
    MaskedAssign( (safe < 0.) , 0., &safety);
    
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void OrbImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
    
    typedef typename Backend::precision_v Double_t;
    Double_t safe(0.);

    Vector3D<Double_t> localPoint;
    localPoint = point;
    
    //General Precalcs
    Double_t rad2    = localPoint.Mag2();
    Double_t rad = Sqrt(rad2);
    safe = unplaced.GetRadius() - rad;
    safety = safe;
    MaskedAssign( (safe < 0.) , 0., &safety);
}


} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
