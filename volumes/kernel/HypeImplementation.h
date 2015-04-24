
/// @file HypeImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedHype.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"

namespace VECGEOM_NAMESPACE { 

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct HypeImplementation {
    
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);
    
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedHype const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedHype const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside);

template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void GenericKernelForContainsAndInside(
    UnplacedHype const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

	typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;	
    
    Float_t dz(unplaced.GetDz());
    Float_t r2 = localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y();
    Float_t radsqIn(0.),radsqOut(0.);
	RadiusHypeSq<Backend,true>(unplaced, localPoint.z(), radsqIn);
	RadiusHypeSq<Backend,false>(unplaced, localPoint.z(), radsqOut);
    if(ForInside)
	completelyinside = (Abs(localPoint.z()) < dz) && (r2 > radsqIn + kTolerance*kTolerance) && (r2 < radsqOut - kTolerance*kTolerance);
    
	completelyoutside =  (Abs(localPoint.z()) > dz) || (r2 < radsqIn - kTolerance*kTolerance) || (r2 > radsqOut + kTolerance*kTolerance);   

}


template <typename Backend,bool ForInnerRad>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void RadiusHypeSq(UnplacedHype const &unplaced, typename Backend::precision_v z, typename Backend::precision_v radsq){
   if(ForInnerRad)
   radsq = unplaced.GetRmin2() + unplaced.GetTIn2()*z*z;
   else
   radsq = unplaced.GetRmax2() + unplaced.GetTOut2()*z*z;
}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedHype const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedHype const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);//{}

 template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedHype const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);//{}
 

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void ContainsKernel(
    UnplacedHype const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside);
  

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void InsideKernel(
    UnplacedHype const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) ;

 
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}


template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}


template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(UnplacedHype const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){
typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;
        safety=0.;
        Float_t safety_t;
        Float_t absZ= Abs(point.z());
        Float_t safeZ= absZ-unplaced.GetDz();
 

        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);
    
        //Outer
        Float_t rhsqOut=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        Float_t rhOut = Sqrt(rhsqOut);
        Float_t drOut = r - rhOut;
        
        Float_t safermax=0.;
        
        Bool_t mask_drOut=(drOut<0);
        MaskedAssign(mask_drOut, -kInfinity, &safermax);
        
        Bool_t mask_fStOut(Abs(unplaced.GetStOut())<kTolerance);
        MaskedAssign(!mask_drOut && mask_fStOut, Abs(drOut), &safermax);
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())*(unplaced.GetTOut2Inv()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;
    
        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        Bool_t doneOuter(mask_fStOut || mask_drOut);
        MaskedAssign(!doneOuter, safe, &safermax);
        Float_t max_safety= Max(safermax, safeZ);
        
        //Check for Inner Threatment -->this should be managed as a specialization
        if(unplaced.GetEndInnerRadius()!=0)
        {
            Float_t safermin=0.;
            Float_t rhsqIn = unplaced.GetRmin2()+unplaced.GetTIn()*point.z()*point.z();
            Float_t rhIn = Sqrt(rhsqIn);
            Float_t drIn = r-rhIn;
            
            Bool_t mask_drIn(drIn>0.);
            MaskedAssign(mask_drIn, -kInfinity, &safermin);
            
            Bool_t mask_fStIn(Abs(unplaced.GetStIn()<kTolerance));
            MaskedAssign(!mask_drIn && mask_fStIn , Abs(drIn), &safermin);
            
            Bool_t mask_fRmin(unplaced.GetRmin()<kTolerance);
            MaskedAssign(! mask_drIn && !mask_fStIn && mask_fRmin, drIn/Sqrt(1.+unplaced.GetTIn2()), &safermin);
            
            Bool_t mask_drMin=Abs(drIn)<kTolerance;
            MaskedAssign(!mask_drIn && !mask_fStIn && !mask_fRmin && mask_drMin, 0., &safermin);
            Bool_t doneInner(mask_drIn || mask_fStIn ||mask_fRmin || mask_drMin );
          
            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) *(unplaced.GetTIn2Inv()) );
            Float_t mIn=-rhIn*unplaced.GetTIn2Inv()/absZ;
            
            safe = mIn*drIn/Sqrt(1.+mIn*mIn);
            MaskedAssign(!doneInner, safe, &safermin);
            max_safety= Max(max_safety, safermin);
        }
        safety=max_safety;
}
  


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(UnplacedHype const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;
        safety=0.;
        
        Float_t absZ= Abs(point.z());
        Float_t safeZ= unplaced.GetDz()-absZ;
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);



        Float_t safermax;
        //OUTER
        Float_t rhsqOut=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        Float_t rhOut = Sqrt(rhsqOut);
        Float_t drOut = r - rhOut;
    
        
        Bool_t mask_fStOut(unplaced.GetStOut()<kTolerance);
        MaskedAssign(mask_fStOut, Abs(drOut), &safermax);
    
        
        Bool_t mask_dr=Abs(drOut)<kTolerance;
        MaskedAssign(!mask_fStOut && mask_dr, 0., &safermax);
        Bool_t doneOuter(mask_fStOut || mask_dr);
        
        
        Float_t mOut= rhOut/(unplaced.GetTOut2()*absZ);
        Float_t saf = -mOut*drOut/Sqrt(1.+mOut*mOut);
        
        MaskedAssign(!doneOuter, saf, &safermax);
        
        safety=Min(safermax, safeZ);
        
        //Check for Inner Threatment
        if(unplaced.GetEndInnerRadius()!=0)
        {
            Float_t safermin=kInfinity;
            Float_t rhsqIn=unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z();
            Float_t rhIn = Sqrt(rhsqIn);
            Float_t drIn = r - rhIn;
        
            Bool_t mask_fStIn(Abs(unplaced.GetStIn())<kTolerance);
            MaskedAssign(mask_fStIn, Abs(drIn), &safermin);
        
            Bool_t mask_fRmin(unplaced.GetRmin()<kTolerance);
            MaskedAssign(!mask_fStIn && mask_fRmin, drIn/Sqrt(1.+unplaced.GetTIn2()), &safermin);
            
            Bool_t mask_drMin=Abs(drIn)<kTolerance;
            MaskedAssign(!mask_fStIn && !mask_fRmin && mask_drMin, 0., &safermin);
        
            Bool_t doneInner(mask_fStIn || mask_fRmin || mask_drMin);
            Bool_t mask_drIn=(drIn<0);
            
            Float_t zHypeSqIn= Sqrt((r*r-unplaced.GetRmin2())/(unplaced.GetTIn2()));
            
            Float_t mIn;
            MaskedAssign(mask_drIn, -rhIn/(unplaced.GetTIn2()*absZ), &mIn);
            MaskedAssign(!mask_drIn, (zHypeSqIn-absZ)/drIn, &mIn);
            
            Float_t safe = mIn*drIn/Sqrt(1.+mIn*mIn);
    
            MaskedAssign(!doneInner, safe, &safermin);
            safety=Min(safety, safermin);
        }
}
  
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Normal(
       UnplacedHype const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );//{}
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedHype const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );//{}
  
  

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void GetHypeRadius2(UnplacedHype const &unplaced,
							typename Backend::precision_v &pZ,
							typename Backend::precision_v &hypeRadius2,bool inner){
		
				typedef typename Backend::precision_v Float_t;
        		typedef typename Backend::bool_v Bool_t;
				
				if(inner)
				return unplaced.GetRmin2() +  unplaced.GetTIn2()*pZ*pZ;
				else
				return unplaced.GetRmax2() +  unplaced.GetTOut2()*pZ*pZ;
}
};


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::Normal(
       UnplacedHype const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){
    //std::cout<<"Entered Normal "<<std::endl;
    NormalKernel<Backend>(unplaced, point, normal, valid);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::NormalKernel(
       UnplacedHype const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){

	//std::cout<<"Entered NormalKernel "<<std::endl;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

	Vector3D<Float_t> localPoint = point;
	Float_t absZ = Abs(localPoint.z());
	Float_t distZ = absZ - unplaced.GetDz();
	Float_t dist2Z = distZ*distZ;
	
	Float_t xR2 = localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y();
	Float_t radOSq(0.);
	RadiusHypeSq(unplaced,localPoint.z(),radOSq,false);
	Float_t dist2Outer = Abs(xR2 - radOSq);
	//std::cout<<"LocalPoint : "<<localPoint<<std::endl;
	Bool_t done(false);
	
	
	//Inner Surface Wins	
	if(unplaced.InnerSurfaceExists())
	{
		Float_t radISq(0.);
		RadiusHypeSq(unplaced,localPoint.z(),radISq,true);
		Float_t dist2Inner = Abs(xR2 - radISq );
		Bool_t cond = (dist2Inner < dist2Z && dist2Inner < dist2Outer);
		MaskedAssign(!done && cond,-localPoint.x(),normal.x());
		MaskedAssign(!done && cond,-localPoint.y(),normal.y());
		MaskedAssign(!done && cond,localPoint.z()*unplaced.GetTIn2(),normal.z());
		normal = normal.Unit();
		done |= cond;
		if(IsFull(done))
			return;
		
	}

	//End Caps wins
	Bool_t condE = (dist2Z < dist2Outer) ;
	Float_t normZ(0.);
	CondAssign(localPoint.z()<0. , -1. ,1. ,normZ);
	MaskedAssign(!done && condE ,0. , normal.x());
	MaskedAssign(!done && condE ,0. , normal.y());
	MaskedAssign(!done && condE ,normZ , normal.z());
	normal = normal.Unit();
	done |= condE;
	if(IsFull(done))
		return;

	//Outer Surface Wins
	normal = Vector3D<Float_t>(localPoint.x(),localPoint.y(),-localPoint.z()*unplaced.GetTOut2()).Unit();
	

}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::UnplacedContains(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside){

      ContainsKernel<Backend>(unplaced, point, inside);
}    

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::Contains(UnplacedHype const &unplaced,
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
void HypeImplementation<transCodeT, rotCodeT>::Inside(UnplacedHype const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside){

    InsideKernel<Backend>(unplaced,
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}
/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::DistanceToIn(UnplacedHype const &unplaced,
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
void HypeImplementation<transCodeT, rotCodeT>::DistanceToOut(UnplacedHype const &unplaced,
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


*/
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void HypeImplementation<transCodeT, rotCodeT>::SafetyToIn(UnplacedHype const &unplaced,
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
void HypeImplementation<transCodeT, rotCodeT>::SafetyToOut(UnplacedHype const &unplaced,
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
void HypeImplementation<transCodeT, rotCodeT>::ContainsKernel(UnplacedHype const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  typedef typename Backend::bool_v Bool_t;
  Bool_t unused;
  Bool_t outside;
  GenericKernelForContainsAndInside<Backend, false>(unplaced,
    localPoint, unused, outside);
  inside=!outside;
}  
/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
UnplacedHype const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

    
    typedef typename Backend::precision_v Float_t;
    //typedef typename Backend::bool_v      Bool_t;	
        
    Precision fR = unplaced.GetRadius();
    Float_t rad2 = localPoint.Mag2();
    
    Float_t tolR = fR - ( kSTolerance*10. );
    if(ForInside)
    completelyinside = (rad2 <= tolR *tolR) ;
    
    tolR = (fR + ( kSTolerance*10.)); 
    completelyoutside = (rad2 >= tolR *tolR);
    //if( IsFull(completelyoutside) )return;

    //Radial Check for GenericKernel Over
}
*/

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::InsideKernel(UnplacedHype const &unplaced,
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
 
/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;
    //typedef typename Backend::inside_v    Inside_t;

    Vector3D<Float_t> localPoint = point;
    Vector3D<Float_t> localDir = direction;
    
    distance = kInfinity;
    Bool_t done(false);
    Bool_t tr(true),fal(false);

    Float_t fR(unplaced.GetRadius()); 
	// General Precalcs
    Float_t rad2 = localPoint.Mag2();
    Float_t rho2 = localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y();
    Float_t pDotV3d = localPoint.Dot(localDir);

    Bool_t cond(false);
  
    Float_t  c(0.), d2(0.);
    c = rad2 - fR * fR;
    //MaskedAssign((tr),(pDotV3d * pDotV3d - c),&d2);
    d2 = (pDotV3d * pDotV3d - c);

    Float_t sd1(kInfinity);
    done |= (d2 < 0. || ((localPoint.Mag() > fR) && (pDotV3d > 0.)));
    if(IsFull(done)) return; //Returning in case of no intersection with outer shell

    MaskedAssign(( (Sqrt(rad2) >= (fR - kSTolerance*10.) ) && (Sqrt(rad2) <= (fR + kSTolerance*10.)) && (pDotV3d < 0.) && !done ),0.,&sd1);
    MaskedAssign( ( (Sqrt(rad2) > (fR + kSTolerance*10.) ) && (tr) && (d2 >= 0.) && pDotV3d < 0.  && !done ) ,(-1.*pDotV3d - Sqrt(d2)),&sd1);
    distance=sd1;


}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

   
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Float_t> localPoint = point;
    Vector3D<Float_t> localDir=direction;
    
    distance = kInfinity;
    Float_t  pDotV2d, pDotV3d;
    
    
    Bool_t done(false);
    Bool_t tr(true),fal(false);
    
    Float_t snxt(kInfinity);
    
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
   
   Bool_t cond1 = (Sqrt(rad2) <= (fR + 0.5*kSTolerance*10.)) ;
   Bool_t cond = (Sqrt(rad2) <= (fR + kSTolerance*10.)) && (Sqrt(rad2) >= (fR - kSTolerance*10.)) && pDotV3d >=0 && cond1;
   done |= cond;
   MaskedAssign(cond ,0.,&sd1);
   
   MaskedAssign((tr && cond1),(pDotV3d * pDotV3d - c),&d2);
   MaskedAssign( (!done && cond1 && (tr) && (d2 >= 0.) ) ,(-1.*pDotV3d + Sqrt(d2)),&sd1);
   
   MaskedAssign((sd1 < 0.),kInfinity, &sd1);
   distance=sd1;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::SafetyToInKernel(UnplacedHype const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){

    typedef typename Backend::precision_v Float_t;
    
    Float_t safe(0.);
    Vector3D<Float_t> localPoint;
    //localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
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
void HypeImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(UnplacedHype const &unplaced,
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
*/


} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
