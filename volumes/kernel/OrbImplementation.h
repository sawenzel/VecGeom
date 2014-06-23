/// @file OrbImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedOrb.h"
#include "base/Vector3D.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct OrbImplementation {

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      typename Backend::int_v &inside) {
    
    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;	
    Bool_t done(false);	

    Double_t rad2 = point.Mag2();
    Double_t tolRMax = unplaced.GetfRTolO();
    Double_t tolRMax2 = tolRMax * tolRMax;
    Double_t tolRMin = unplaced.GetfRTolI();
    Double_t tolRMin2 = tolRMin * tolRMin;

    //Point is Outside
    Bool_t isOutside = ( rad2 > tolRMax2);
    done |= isOutside;
    MaskedAssign(isOutside, EInside::kOutside, &inside);
    if(done == Backend::kTrue) return;

    //Point is Inside
    Bool_t isInside = ( rad2 < tolRMax2);
    done |= isInside;
    MaskedAssign(isInside, EInside::kInside, &inside);
    if(done == Backend::kTrue) return;
    
    //Point is on the Surface
    MaskedAssign(!done, EInside::kSurface, &inside);
    return;
  }


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside){

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;	
	
    Bool_t done(false);
    Double_t rad2 = point.Mag2();

    Double_t tolRMax = unplaced.GetfRTolO();
    Double_t tolRMax2 = tolRMax * tolRMax;
    Double_t tolRMin = unplaced.GetfRTolI();
    Double_t tolRMin2 = tolRMin * tolRMin;

    Bool_t inSide(Backend::kTrue);

    //Point is Outside
    Bool_t isOutside = ( rad2 > tolRMax2);
    done=isOutside;
    MaskedAssign(isOutside, !inSide, &inside);
    if(done == Backend::kTrue)return;

    //Point is Inside    
    Bool_t isInside = ( rad2 < tolRMin2);
    done |= isInside;
    MaskedAssign(isInside, inSide, &inside);
    if(done == Backend::kTrue)return;

    //Point is on the Surface
    MaskedAssign(!done, !inSide, &inside);
    return;
}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedOrb const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside){

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(unplaced, localPoint, inside);
}


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedOrb const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside){

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;  

    Vector3D<Double_t> localPoint;
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    InsideKernel<Backend>(unplaced, localPoint, inside);
}



  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedOrb const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Double_t> localPoint;
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);

    Vector3D<Double_t> localDir;
    localDir =  transformation.TransformDirection<rotCodeT>(direction);


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
    if(done == Backend::kTrue)return;

 
    Bool_t notOutsideAndOnSurface = (c > (-kTolerance * unplaced.GetRadius()));
    Bool_t d2LTFrTolFrAndPDotV3DGTET0= ((d2 < (kTolerance * unplaced.GetRadius())) || (pDotV3d >= 0));
    done |= (notOutsideAndOnSurface && d2LTFrTolFrAndPDotV3DGTET0);
    if(done == Backend::kTrue) return;


    Bool_t isOutsideTolBoundary = (c > (kTolerance * unplaced.GetRadius()));
    Bool_t isD2GtEt0 = (d2>=0);
    done |= (isOutsideTolBoundary && !isD2GtEt0);
    if(done == Backend::kTrue) return;

    done |= (isOutsideTolBoundary && isD2GtEt0 );
    MaskedAssign((isOutsideTolBoundary && isD2GtEt0),(-pDotV3d - Sqrt(d2)),&distance);
    //(distance < 0 ) implies point is outside and going out, hence distance must be set to kInfinity.
    MaskedAssign((distance<0),kInfinity,&distance);
    if(done == Backend::kTrue) return;

}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

    distance = kInfinity;  
    Double_t zero=Backend::kZero;

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

    //checking if the point is outside
    Double_t tolRMax = unplaced.GetfRTolO();
    Double_t tolRMax2 = tolRMax * tolRMax;
    Bool_t isOutside = ( rad2 > tolRMax2);
    done|= isOutside;
    if (done == Backend::kTrue) return;

    Bool_t isInsideAndWithinOuterTolerance = ((rad <= tolRMax) && (c < (kTolerance * unplaced.GetRadius())));
    Bool_t isInsideAndOnTolerantSurface = ((c > (-2*kTolerance*unplaced.GetRadius())) && ( (pDotV3d >= 0) || (d2 < 0) ));

    Bool_t onSurface=(isInsideAndWithinOuterTolerance && isInsideAndOnTolerantSurface );
    MaskedAssign(onSurface , zero, &distance);
    done|=onSurface;
    if (done == Backend::kTrue) return;

    Bool_t notOnSurface=(isInsideAndWithinOuterTolerance && !isInsideAndOnTolerantSurface );
    MaskedAssign(notOnSurface , (-pDotV3d + Sqrt(d2)), &distance);
    done|=notOnSurface;
    if (done == Backend::kTrue) return;
    
}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedOrb const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

    Double_t safe=Backend::kZero;
    Double_t zero=Backend::kZero; 

    Vector3D<Double_t> localPoint;
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);

    //General Precalcs
    Double_t rad2    = localPoint.Mag2();
    Double_t rad = Sqrt(rad2);
    safe = rad - unplaced.GetRadius();
    safety = safe;
    MaskedAssign( (safe < zero) , zero, &safety);
    
}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

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

};


} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
