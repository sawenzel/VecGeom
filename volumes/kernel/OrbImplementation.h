/// @file OrbImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
//#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedOrb.h"

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
    Bool_t isInside = ( radius2 < tolRMax2);
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
    Vector3D<typename Backend::precision_v> localPoint;
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
      typename Backend::precision_v &distance){}

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
