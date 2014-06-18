/// @file OrbImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
//#include "volumes/kernel/GenericKernels.h"
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
	
	Double_t radius2 = point.x()*point.x() + point.y()*point.y() + point.z()*point.z();
    Double_t tolRMax = unplaced.GetfRTolO();
	Double_t tolRMax2 = tolRMax * tolRMax;

	Bool_t isOutside = ( radius2 > tolRMax2);
	Bool_t done(isOutside);
    MaskedAssign(isOutside, EInside::kOutside, &inside);
    if(done == Backend::kTrue) return;

    Double_t tolRMin = unplaced.GetfRTolI();
	Double_t tolRMin2 = tolRMin * tolRMin;
    Bool_t isInside = ( radius2 < tolRMax2);
    MaskedAssign(isInside, EInside::kInside, &inside);
    done |= isInside;
	if(done == Backend::kTrue) return;
    
    MaskedAssign(!done, EInside::kSurface, &inside);
 
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
	
	
	Double_t radius2 = point.x()*point.x() + point.y()*point.y() + point.z()*point.z();
    Double_t tolRMax = unplaced.GetfRTolO();
	Double_t tolRMax2 = tolRMax * tolRMax;

	Bool_t isOutside = ( radius2 > tolRMax2);
	Bool_t done(isOutside);
    //MaskedAssign(isOutside, EInside::kOutside, &inside);
	//MaskedAssign(isOutside, 0., &inside);
    if(done == Backend::kTrue) 
	{	inside=Backend::kFalse;//EInside::kOutside;
		return;
    }

    Double_t tolRMin = unplaced.GetfRTolI();
	Double_t tolRMin2 = tolRMin * tolRMin;
    Bool_t isInside = ( radius2 < tolRMin2);
    //MaskedAssign(isInside, EInside::kInside, &inside);
	//MaskedAssign(isOutside, 1., &inside);
    done |= isInside;
	if(done == Backend::kTrue)
	{
		inside=Backend::kTrue;//EInside::kInside;
	return;
	}
    
    //MaskedAssign(!done, EInside::kSurface, &inside);
	//MaskedAssign(!done, 2., &inside);
	if(!done)
		{
		inside=Backend::kFalse;//EInside::kSurface;
		}
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

  //  Vector3D<typename Backend::precision_v> localPoint;
  //  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);

    typedef typename Backend::precision_v Double_t;
    typedef typename Backend::bool_v      Bool_t;

    distance = kInfinity;  
	Double_t zero=Backend::kZero;

    Vector3D<Double_t> localPoint;
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);

    Vector3D<Double_t> localDir;
    localDir =  transformation.Transform<transCodeT, rotCodeT>(direction);

    //General Precalcs
    Double_t rad2    = (localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y() + localPoint.z() * localPoint.z());
    Double_t rad = sqrt(rad2);
    Double_t pDotV3d = localPoint.x() * localDir.x() + localPoint.y() * localDir.y() + localPoint.z() * localDir.z();
    Double_t radius2 = unplaced.GetRadius() * unplaced.GetRadius();
    Double_t c = rad2 - radius2;
    Double_t d2 = pDotV3d * pDotV3d - c;

    Bool_t done(false);
    distance = kInfinity;

	Double_t pos_dot_dir_x = localPoint.x()*localDir.x();
    Double_t pos_dot_dir_y = localPoint.y()*localDir.y();
	Double_t pos_dot_dir_z = localPoint.z()*localDir.z();

    // outside of sphere and going away?
    //check if the point is distancing in X
    Bool_t isDistancingInX = ( Abs(localPoint.x()) > unplaced.GetRadius() ) && (pos_dot_dir_x > 0);
    done|=isDistancingInX;
	if (done == Backend::kTrue) return;

    //check if the point is distancing in Y
    Bool_t isDistancingInY = ( Abs(localPoint.y()) > unplaced.GetRadius() ) && (pos_dot_dir_y > 0);
    done|=isDistancingInY;
    if (done == Backend::kTrue) return;

    //check if the point is distancing in Z
    Bool_t isDistancingInZ = ( Abs(localPoint.z()) > unplaced.GetRadius() ) && (pos_dot_dir_z > 0);
    done|=isDistancingInZ;
    if (done == Backend::kTrue) return;

    //checking if the poing is inside
    Double_t tolRMin = unplaced.GetfRTolI();
	Double_t tolRMin2 = tolRMin * tolRMin;
    Bool_t isInside = ( rad2 < tolRMin2);
    done|= isInside;
    if (done == Backend::kTrue) return;


    Bool_t notInsideButOutsideTolerantBoundary=((rad > (unplaced.GetRadius() - kHalfTolerance)) && (c > (kTolerance * unplaced.GetRadius())) && (d2 >= 0));
	Double_t s = -pDotV3d - Sqrt(d2);
    MaskedAssign(notInsideButOutsideTolerantBoundary , s, &distance);
    done|=notInsideButOutsideTolerantBoundary;
	if (done == Backend::kTrue) return;

	
    Bool_t notInsideButInsideTolerantBoundary=((rad > (unplaced.GetRadius() - kHalfTolerance)) && (c > (-kTolerance * unplaced.GetRadius())) && ((d2 < (kTolerance * unplaced.GetRadius())) || (pDotV3d >= 0)));
//	MaskedAssign(notInsideButInsideTolerantBoundary , zero, &distance);
    done|=notInsideButInsideTolerantBoundary;
	if (done == Backend::kTrue) return;
    
    MaskedAssign(!done , zero, &distance);

}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedOrb const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedOrb const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedOrb const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){}

};


} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
