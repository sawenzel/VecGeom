
/// @file SphereImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
//#include "math.h"
#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedSphere.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
//#include <iomanip>
//#include <Vc/Vc>
//#include "TGeoShape.h"
//#include "volumes/SphereUtilities.h"

#include <stdio.h>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(SphereImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE { 
 
class PlacedSphere;
 
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct SphereImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

using PlacedShape_t = PlacedSphere;
using UnplacedShape_t = UnplacedSphere;

VECGEOM_CUDA_HEADER_BOTH
static void PrintType() {
   printf("SpecializedSphere<%i, %i>", transCodeT, rotCodeT);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE    
static typename Backend::precision_v fabs(typename Backend::precision_v &v)
  {
      typedef typename Backend::precision_v Float_t;
      Float_t mone(-1.);
      Float_t ret(0);
      MaskedAssign( (v<0), mone*v , &ret );
      return ret;
  }

    
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);//{}
    
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedSphere const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);//{}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedSphere const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside);//{}

template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void GenericKernelForContainsAndInside(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside);// {}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedSphere const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);//{}

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);//{}



template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedSphere const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);//{}

 template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);//{}
 

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void ContainsKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside);//{}
  

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void InsideKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside);// {}

 
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){}



template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(UnplacedSphere const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);//{}
  


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);//{}
  
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Normal(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );//{}
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );//{}
  
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ApproxSurfaceNormalKernel(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal);
  
  //Trying to follow Distance Algo from ROOT
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToSphere(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
  	  typename Backend::precision_v const &radius,
      typename Backend::bool_v &check,
	  typename Backend::bool_v &firstcross,
      typename Backend::precision_v &distance);//{}  
  
    
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToCone(
     UnplacedSphere const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> const &direction,
     typename Backend::precision_v const &dzv,
     typename Backend::precision_v const &r1v,
     typename Backend::precision_v const &r2v,
     typename Backend::precision_v &b1v,
     typename Backend::precision_v &deltav
  );
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToPhiMin(
     UnplacedSphere const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> const &direction,
     typename Backend::precision_v const &s1,
     typename Backend::precision_v const &c1,
     typename Backend::precision_v const &s2,
     typename Backend::precision_v const &c2,
     typename Backend::precision_v const &sm,
     typename Backend::precision_v const &cm,
     typename Backend::bool_v &in,
          typename Backend::precision_v &distance
  );
  
  
  
};

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToPhiMin(
     UnplacedSphere const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> const &direction,
     typename Backend::precision_v const &s1,
     typename Backend::precision_v const &c1,
     typename Backend::precision_v const &s2,
     typename Backend::precision_v const &c2,
     typename Backend::precision_v const &sm,
     typename Backend::precision_v const &cm,
     typename Backend::bool_v &in,
        typename Backend::precision_v &distance){
    
  //std::cout<<"Entered DistanceTo_PHI_MIN"<<std::endl;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    
    Float_t zero=Backend::kZero;
    //distance = zero;
    
    Vector3D<Float_t> localPoint;
    localPoint = point;
    
    Vector3D<Float_t> localDir;
    localDir =  direction;
    
    Float_t mone(-1);
    Bool_t tr(true);

    Float_t sfi1(kInfinity);
    Float_t sfi2(kInfinity);
    Float_t s(zero);
    Float_t un = localDir.x()*s1 - localDir.y()*c1;
    MaskedAssign((!in),mone*un,&un);
    MaskedAssign((un > zero),(mone*localPoint.x()*s1 + localPoint.y()*c1),&s);
    MaskedAssign(( (un > zero) && (!in) ),mone*s,&s);
    MaskedAssign(( (un > zero) && (s >= zero)),(s/un),&s);
    MaskedAssign(( (un > zero) && (s >= zero) && (((localPoint.x()+s*localDir.x())*sm-(localPoint.y()+s*localDir.y())*cm)>=zero ) ),s,&sfi1);
    
    
    un = mone*localDir.x()*s2 + localDir.y()*c2;
    MaskedAssign((!in),mone*un,&un);
    MaskedAssign((un>zero) ,(localPoint.x()*s2-localPoint.y()*c2) , &s);
    MaskedAssign(((un>zero) && (!in) ),mone*s,&s);
    MaskedAssign(( (un > zero) && (s >= zero)),(s/un),&s);
    MaskedAssign(( (un > zero) && (s >= zero) && ((mone*(localPoint.x()+s*localDir.x())*sm+(localPoint.y()+s*localDir.y())*cm)>=zero)),s,&sfi2);
    
    //std::cout<<"SF1 : "<<sfi1<<"  :: SF2 : "<<sfi1<<std::endl;
    
    MaskedAssign(tr,Min(sfi1,sfi2),&distance);
    
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToCone(
     UnplacedSphere const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> const &direction,
     typename Backend::precision_v const &dzv,
     typename Backend::precision_v const &r1v,
     typename Backend::precision_v const &r2v,
     typename Backend::precision_v &bv,
     typename Backend::precision_v &deltav){
    
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Bool_t done(false);
    Float_t zero(0.);
    Float_t mone(-1.);
    Float_t one(1.);
    
    Vector3D<Float_t> localPoint;
    localPoint = point;

    Vector3D<Float_t> localDir;
    localDir = direction;
    
    //General Precalcs
    Float_t rad2 = localPoint.Mag2(); //r2
    Float_t rad = Sqrt(rad2);
    Float_t pDotV3d = localPoint.Dot(localDir); //b

    //Starting logic to Calculate DistanceToCone
    deltav = mone;
    done |= (dzv < zero);
    if( IsFull(done) )return ;
    
    Float_t ro0v = (0.5 *(r1v + r2v));
    Float_t tzv = (0.5 * (r2v -r1v)/dzv);
    Float_t rsqv = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rcv = ro0v + localPoint.z()*tzv;
    
    Float_t av = localDir.x() * localDir.x() + localDir.y() * localDir.y() - tzv*tzv*localDir.z() * localDir.z();
    bv = localPoint.x() * localDir.x() + localPoint.y() * localDir.y() - tzv*rcv*localDir.z();
    Float_t cv = rsqv - rcv*rcv;
    
    Float_t temp = av; Float_t temp2 = bv;
    MaskedAssign( (temp < zero) , (mone*temp), &temp);
    MaskedAssign( (temp2 < zero) , (mone*temp2), &temp2);
    
    Float_t toler(1.e-10);
    done |= ((temp < toler) && (temp2 < toler));
    if( IsFull(done) )return ;
    
    done |= (temp < toler);
    MaskedAssign( done , (0.5 * (cv/bv)),&bv);
    MaskedAssign( done, zero , &deltav);
    if( IsFull(done) )return ;
    
    av = (one/av);
    bv *= av;
    cv *= av;
    deltav = bv*bv - cv;
    
    MaskedAssign( (deltav > zero) , Sqrt(deltav), &deltav);
    MaskedAssign( !(deltav > zero) , mone , &deltav );
    
    
  }

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToSphere(
//typename Backend::precision_v SphereImplementation<transCodeT, rotCodeT>::DistanceToSphere(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
	  typename Backend::precision_v const &radius,
      typename Backend::bool_v &check,
	  typename Backend::bool_v &firstcross, 
      typename Backend::precision_v &distance){
  
  // bool verbose=false;
	    //std::cout<<"Raman insdie "<<std::endl;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Bool_t done(false);
    Float_t zero(0.);
    Float_t mone(-1.);
    Vector3D<Float_t> localPoint;
    localPoint = point;

    Vector3D<Float_t> localDir;
    localDir = direction;

    //General Precalcs
    Float_t rad2 = localPoint.Mag2(); //r2
    Float_t rad = Sqrt(rad2);
    Float_t pDotV3d = localPoint.Dot(localDir); //b

    Float_t radius2 = radius * radius; //radius = rsph (in ROOT)
    Float_t c = rad2 - radius2;
    Float_t d2 = pDotV3d * pDotV3d - c;
	Float_t d = Sqrt(d2);
   //std::cout<<"D = "<<d<<std::endl;
	done |= (d2 <= zero);
	MaskedAssign( (d2 <= zero) , kInfinity, &distance );
	if( IsFull(done) )return ;

	Vector3D<Float_t> pt;
	//Bool_t in(false);
 	//MaskedAssign( (c <= zero) , true , &in);
	Bool_t in = (c <= zero);

	MaskedAssign((in) ,((mone * pDotV3d) + d)  , &distance);
	MaskedAssign( ((!in) && (firstcross)) ,((mone * pDotV3d) - d)  , &distance);
	MaskedAssign( ((!in) && (!firstcross)) ,((mone * pDotV3d) + d)  , &distance);

	MaskedAssign( (distance  < zero) , kInfinity  , &distance );
	
	
	pt =  localPoint + (distance * localDir);
        Bool_t inside(true);
	ContainsKernel<Backend>(unplaced,pt,inside);
	MaskedAssign( ( (check) && !inside ), kInfinity , &distance );
	//MaskedAssign( ( (check)  ) , kInfinity , &distance );
	//if(verbose)std::cout<<"From DistanceToSphere : "<<distance<<std::endl;
	

}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::Normal(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){

    NormalKernel<Backend>(unplaced, point, normal, valid);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::ApproxSurfaceNormalKernel(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &norm){

    //std::cout<<"Entered AppoxNormalKernel"<<std::endl;
        
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;
    
    Float_t kNRMin(0.), kNRMax(1.), kNSPhi(2.), kNEPhi(3.), kNSTheta(4.), kNETheta(5.);
    Float_t side(10.);
 
    Float_t rho, rho2, radius;
    Float_t distRMin(0.), distRMax(0.), distSPhi(0.), distEPhi(0.), distSTheta(0.), distETheta(0.), distMin(0.);
    Float_t zero(0.),mone(-1.);
    Float_t temp=zero;
    
    Vector3D<Float_t> localPoint;
    localPoint = point;
    
    Float_t radius2=localPoint.Mag2();
    radius = Sqrt(radius2);
    rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    rho = Sqrt(rho2);
    
    Float_t fRmax = unplaced.GetOuterRadius();
    Float_t fRmin = unplaced.GetInnerRadius();
    Float_t fSPhi = unplaced.GetStartPhiAngle();
    Float_t fDPhi = unplaced.GetDeltaPhiAngle();
    Float_t ePhi = fSPhi + fDPhi;
    Float_t fSTheta = unplaced.GetStartThetaAngle();
    Float_t fDTheta = unplaced.GetDeltaThetaAngle();
    Float_t eTheta = fSTheta + fDTheta;
    Float_t pPhi = localPoint.Phi();
    Float_t pTheta(std::atan2(rho,localPoint.z()));
    Float_t sinSPhi = std::sin(fSPhi);
    Float_t cosSPhi = std::cos(fSPhi);
    Float_t sinEPhi = std::sin(ePhi);
    Float_t cosEPhi = std::cos(ePhi);
    Float_t sinSTheta = std::sin(fSTheta);
    Float_t cosSTheta = std::cos(fSTheta);
    Float_t sinETheta = std::sin(eTheta);
    Float_t cosETheta = std::cos(eTheta);
        
    //
    // Distance to r shells
    //
    temp = radius - fRmax;
    MaskedAssign((temp<zero),mone*temp,&temp);
    distRMax = temp;
    
    temp = radius - fRmin;
    MaskedAssign((temp<zero),mone*temp,&temp);
    distRMin = temp;
    Float_t prevDistMin(zero);
    prevDistMin = distMin;
    MaskedAssign( ( (fRmin > zero) && (distRMin < distRMax) ) , distRMin, &distMin );
    MaskedAssign( ( (fRmin > zero) && (distRMin < distRMax) ) , kNRMin, &side ); //ENorm issue : Resolved hopefully
    
    prevDistMin = distMin;
    MaskedAssign( ( (fRmin > zero) && !(distRMin < distRMax) ) , distRMax, &distMin );
    MaskedAssign( ( (fRmin > zero) && !(distRMin < distRMax) ) , kNRMax, &side );//ENorm issue : Resolved hopefully
    
    MaskedAssign( !(fRmin > zero), distRMax, &distMin );
    MaskedAssign( !(fRmin > zero), kNRMax, &side );
    
    //
    // Distance to phi planes
    //
    // Protected against (0,0,z)
    
    MaskedAssign( (pPhi < zero) ,(pPhi+(2*kPi)) , &pPhi);
    
    temp = pPhi - (fSPhi + 2 * kPi);
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) && (fSPhi < zero) ) ,(temp*rho)  ,&distSPhi);
    
    temp = pPhi - fSPhi;
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) && !(fSPhi < zero) ) ,(temp*rho)  ,&distSPhi);
    
    temp=pPhi - fSPhi - fDPhi;
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign((!unplaced.IsFullPhiSphere() && (rho>zero)), temp*rho, &distEPhi); //distEPhi = temp * rho;
    
    
    prevDistMin = distMin;
    //std::cout<<"--------------------------------------------------------------------------------"<<std::endl;
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && (distSPhi < distEPhi) && (distSPhi < distMin)),distSPhi ,&distMin );
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && (distSPhi < distEPhi) && (distSPhi < prevDistMin)),kNSPhi ,&side ); //CULPRIT
    
    prevDistMin = distMin;
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && !(distSPhi < distEPhi) && (distEPhi < distMin)),distEPhi ,&distMin );
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && !(distSPhi < distEPhi) && (distEPhi < prevDistMin)),kNEPhi ,&side );
    
    //
    // Distance to theta planes
    //
    temp = pTheta - fSTheta;
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero)) , temp*radius ,&distSTheta);
    
    temp = pTheta - fSTheta - fDTheta;
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero)) , temp*radius ,&distETheta);
    
    prevDistMin = distMin;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && (distSTheta < distETheta) && (distSTheta < distMin)), distSTheta, &distMin); 
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && (distSTheta < distETheta) && (distSTheta < prevDistMin)), kNSTheta, &side); 
    //std::cout<<" (distSTheta < distMin) : "<<distSTheta<<" , "<<distETheta<<" :: "<<distMin<<" :: side :"<<side<<std::endl;
    
    prevDistMin = distMin;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && !(distSTheta < distETheta) && (distETheta < distMin)), distETheta, &distMin); 
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && !(distSTheta < distETheta) && (distETheta < prevDistMin)), kNETheta, &side);
    // std::cout<<" !(distSTheta < distMin) : "<<distSTheta<<" , "<<distETheta<<" :: "<<distMin<<" :: side :"<<side<<std::endl;
    
    //Switching
    //std::cout<<"Side : "<<side<<std::endl;
    Bool_t done(false);
    done |= (side == kNRMin);
    MaskedAssign( (side == kNRMin), Vector3D<Float_t>(-localPoint.x() / radius, -localPoint.y() / radius, -localPoint.z() / radius),&norm);
   
    if( IsFull(done) )return ;//{  std::cout<<"----- 1 ------"<<std::endl; return;}
    
    done |= (side == kNRMax);
    MaskedAssign( (side == kNRMax),Vector3D<Float_t>(localPoint.x() / radius, localPoint.y() / radius, localPoint.z() / radius),&norm);
    //std::cout<<"----- 2 ------"<<std::endl;
    if( IsFull(done) )return ;//{  std::cout<<"----- 2 ------"<<std::endl; return;}
    
    done |= (side == kNSPhi);
    MaskedAssign( (side == kNSPhi),Vector3D<Float_t>(sinSPhi, -cosSPhi, 0),&norm);
    //std::cout<<"----- 3 ------"<<std::endl;
    if( IsFull(done) )return ;//{  std::cout<<"----- 3 ------"<<std::endl; return;}
    
    done |= (side == kNEPhi);
    MaskedAssign( (side == kNEPhi),Vector3D<Float_t>(-sinEPhi, cosEPhi, 0),&norm);
    //std::cout<<"----- 4 ------"<<std::endl;
    if( IsFull(done) )return ;//{  std::cout<<"----- 4 ------"<<std::endl; return;}
    
    done |= (side == kNSTheta);
    MaskedAssign( (side == kNSTheta),Vector3D<Float_t>(-cosSTheta * std::cos(pPhi), -cosSTheta * std::sin(pPhi),sinSTheta),&norm);
    //std::cout<<"----- 5 ------"<<std::endl;
    if( IsFull(done) )return ;//{  std::cout<<"----- 5 ------"<<std::endl; return;}
    
    done |= (side == kNETheta);
    MaskedAssign( (side == kNETheta),Vector3D<Float_t>(cosETheta * std::cos(pPhi), cosETheta * std::sin(pPhi), sinETheta),&norm);
    //std::cout<<"----- 6 ------"<<std::endl;
    if( IsFull(done) )return ;//{  std::cout<<"----- 6 ------"<<std::endl; return;}
    
       
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::NormalKernel(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Float_t> localPoint;
    localPoint = point;
    
    Float_t radius2=localPoint.Mag2();
    Float_t radius=Sqrt(radius2);
    Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rho = Sqrt(rho2);
    Float_t fRmax = unplaced.GetOuterRadius();
    Float_t fRmin = unplaced.GetInnerRadius();
    Float_t fSPhi = unplaced.GetStartPhiAngle();
    Float_t fDPhi = unplaced.GetDeltaPhiAngle();
    Float_t ePhi = fSPhi + fDPhi;
    Float_t fSTheta = unplaced.GetStartThetaAngle();
    Float_t fDTheta = unplaced.GetDeltaThetaAngle();
    Float_t eTheta = fSTheta + fDTheta;
    Float_t pPhi = localPoint.Phi();
    Float_t pTheta(std::atan2(rho,localPoint.z()));
    Float_t sinSPhi = std::sin(fSPhi);
    Float_t cosSPhi = std::cos(fSPhi);
    Float_t sinEPhi = std::sin(ePhi);
    Float_t cosEPhi = std::cos(ePhi);
    Float_t sinSTheta = std::sin(fSTheta);
    Float_t cosSTheta = std::cos(fSTheta);
    Float_t sinETheta = std::sin(eTheta);
    Float_t cosETheta = std::cos(eTheta);
    
    //Precision fRminTolerance = unplaced.GetFRminTolerance();
    Precision kAngTolerance = unplaced.GetAngTolerance();
    Precision halfAngTolerance = (0.5 * kAngTolerance);
    
    Float_t distSPhi(kInfinity),distSTheta(kInfinity);
    Float_t distEPhi(kInfinity),distETheta(kInfinity);
    Float_t distRMax(kInfinity); 
    Float_t distRMin(kInfinity);
    
    Vector3D<Float_t> nR, nPs, nPe, nTs, nTe, nZ(0., 0., 1.);
    Vector3D<Float_t> norm, sumnorm(0., 0., 0.);
    
    Bool_t fFullPhiSphere = unplaced.IsFullPhiSphere();
    //std::cout<<"Full Phi Sphere Check : "<<fFullPhiSphere<<std::endl;
    Bool_t fFullThetaSphere = unplaced.IsFullThetaSphere();
    //std::cout<<"Full Theta Sphere Check : "<<fFullThetaSphere<<std::endl;

    Float_t zero(0.);
    Float_t mone(-1.);
    
    Float_t temp=0;
    temp=radius - fRmax;
    MaskedAssign((temp<zero),mone*temp,&temp);
    //distRMax = fabs<Backend>(temp);
    distRMax = temp;
    
    distRMin = 0;
    temp = radius - fRmin;
    MaskedAssign((temp<zero),mone*temp,&temp);
    //MaskedAssign( (fRmin > 0) , fabs<Backend>(temp), &distRMin);
    MaskedAssign( (fRmin > 0) , temp, &distRMin);
    
    MaskedAssign( ((rho>zero) && !(unplaced.IsFullSphere()) && (pPhi < fSPhi - halfAngTolerance)) , (pPhi+(2*kPi)), &pPhi);
    MaskedAssign( ((rho>zero) && !(unplaced.IsFullSphere()) && (pPhi > ePhi + halfAngTolerance)) , (pPhi-(2*kPi)), &pPhi);
    
    //Phi Stuff
    temp = pPhi - fSPhi;
    MaskedAssign((temp<zero),mone*temp,&temp);
    //MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) ), (fabs<Backend>(temp)),&distSPhi);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) ), temp,&distSPhi);
    temp = pPhi - ePhi;
    MaskedAssign((temp<zero),mone*temp,&temp);
    //MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) ), (fabs<Backend>(temp)),&distEPhi);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) ), temp,&distEPhi);
    
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (!fRmin) ), zero ,&distSPhi);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (!fRmin) ), zero ,&distEPhi);
    MaskedAssign( (!unplaced.IsFullPhiSphere()), Vector3D<Float_t>(sinSPhi, -cosSPhi, 0),&nPs );
    MaskedAssign( (!unplaced.IsFullPhiSphere()), Vector3D<Float_t>(-sinEPhi, cosEPhi, 0),&nPe );
    
    //Theta Stuff
    temp = pTheta - fSTheta;
    MaskedAssign((temp<zero),mone*temp,&temp);
    //MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho>zero) ) , fabs<Backend>(temp) , &distSTheta ) ;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho>zero) ) , temp , &distSTheta ) ;
    
    temp = pTheta - eTheta;
    MaskedAssign((temp<zero),mone*temp,&temp);
    //MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho>zero) ) , fabs<Backend>(temp) , &distETheta ) ;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho>zero) ) , temp , &distETheta ) ;
    
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho>zero) ) , Vector3D<Float_t>(-cosSTheta * localPoint.x() / rho , -cosSTheta * localPoint.y() / rho, sinSTheta ) , &nTs ) ;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho) ) , Vector3D<Float_t>(cosETheta * localPoint.x() / rho , cosETheta * localPoint.y() / rho , -sinETheta) , &nTe ) ;
    
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (!fRmin) && (fSTheta)) , zero , &distSTheta );
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (!fRmin) && (fSTheta)) , Vector3D<Float_t>(0., 0., -1.) , &nTs );
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (!fRmin) && (eTheta < kPi)) , zero , &distETheta );
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (!fRmin) && (eTheta < kPi)) , Vector3D<Float_t>(0., 0., 1.) , &nTe );
    
    MaskedAssign( (radius), Vector3D<Float_t>(localPoint.x() / radius, localPoint.y() / radius, localPoint.z() / radius) ,&nR);
    
    
    Float_t noSurfaces(0);
    Float_t halfCarTolerance(0.5 * 1e-9);
    MaskedAssign((distRMax <= halfCarTolerance) , noSurfaces+1 ,&noSurfaces);
    MaskedAssign((distRMax <= halfCarTolerance) , (sumnorm+nR) ,&sumnorm);
    
    MaskedAssign((fRmin && (distRMin <= halfCarTolerance)) , noSurfaces+1 ,&noSurfaces);
    MaskedAssign((fRmin && (distRMin <= halfCarTolerance)) , (sumnorm-nR) ,&sumnorm);
    
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (distSPhi <= halfAngTolerance)) , noSurfaces+1 ,&noSurfaces);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (distSPhi <= halfAngTolerance)) , (sumnorm+nPs) ,&sumnorm);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (distEPhi <= halfAngTolerance)) , noSurfaces+1 ,&noSurfaces);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (distEPhi <= halfAngTolerance)) , (sumnorm+nPe) ,&sumnorm);
    
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distSTheta <= halfAngTolerance) && (fSTheta > zero)), noSurfaces+1 ,&noSurfaces);
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distSTheta <= halfAngTolerance) && (fSTheta > zero) && ((radius <= halfCarTolerance) && fFullPhiSphere)), (sumnorm+nZ) ,&sumnorm);
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distSTheta <= halfAngTolerance) && (fSTheta > zero) && !((radius <= halfCarTolerance) && fFullPhiSphere)), (sumnorm+nTs) ,&sumnorm);
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distETheta <= halfAngTolerance) && (eTheta < kPi)), noSurfaces+1 ,&noSurfaces);
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distETheta <= halfAngTolerance) && (eTheta < kPi) && ((radius <= halfCarTolerance) && fFullPhiSphere)), (sumnorm-nZ) ,&sumnorm);
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distETheta <= halfAngTolerance) && (eTheta < kPi) && !((radius <= halfCarTolerance) && fFullPhiSphere)), (sumnorm+nTe) ,&sumnorm);
    MaskedAssign( (!unplaced.IsFullThetaSphere() && (distETheta <= halfAngTolerance) && (eTheta < kPi) && (sumnorm.z() == zero)), (sumnorm+nZ) ,&sumnorm);
    
    //Now considering case of ApproxSurfaceNormal
    if(noSurfaces == 0)
        ApproxSurfaceNormalKernel<Backend>(unplaced,point,norm);
    
    MaskedAssign((noSurfaces == 1),sumnorm,&norm);
    MaskedAssign((!(noSurfaces == 1) && (noSurfaces !=0 )),(sumnorm*1./sumnorm.Mag()),&norm);
    MaskedAssign(true,norm,&normal);
    
    
    valid = (noSurfaces>zero);
   
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::Contains(UnplacedSphere const &unplaced,
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
void SphereImplementation<transCodeT, rotCodeT>::UnplacedContains(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside){

      ContainsKernel<Backend>(unplaced, point, inside);
}    


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::ContainsKernel(UnplacedSphere const &unplaced,
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
void SphereImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v      Bool_t;	

    Precision fRmin = unplaced.GetInnerRadius();
    Precision fRminTolerance = unplaced.GetFRminTolerance();
    Precision kAngTolerance = unplaced.GetAngTolerance();
    Precision halfAngTolerance = (0.5 * kAngTolerance);
    Precision fRmax = unplaced.GetOuterRadius();
        
    Float_t rad2 = localPoint.Mag2();
    Float_t tolRMin = fRmin + (0.5 * fRminTolerance); //rMinPlus
    // Float_t tolRMin2 = tolRMin * tolRMin;
    Float_t tolRMax = fRmax - (0.5 * fRminTolerance); //rMaxMinus
    // Float_t tolRMax2 = tolRMax * tolRMax;
     
    // Check radial surfaces
    //Radial check for GenericKernel Start
    completelyinside = (rad2 <= tolRMax*tolRMax) && (rad2 >= tolRMin*tolRMin);
    
    tolRMin = fRmin - (0.5 * fRminTolerance); //rMinMinus
    tolRMax = fRmax + (0.5 * fRminTolerance); //rMaxPlus
    
    completelyoutside = (rad2 <= tolRMin*tolRMin) || (rad2 >= tolRMax*tolRMax);
    if( IsFull(completelyoutside) )return;
    //return; //Later on remove it and should be only at the end when checks for PHI and THETA finishes
    
    //Radial Check for GenericKernel Over
    
    
    
    Float_t pPhi = localPoint.Phi();
    Float_t fSPhi(unplaced.GetStartPhiAngle());
    Float_t fDPhi(unplaced.GetDeltaPhiAngle());
    Float_t ePhi = fSPhi+fDPhi;
    
    //*******************************
    //Very important but needs to understand
    MaskedAssign((pPhi<(fSPhi - halfAngTolerance)),pPhi+(2.*kPi),&pPhi);
    MaskedAssign((pPhi>(ePhi + halfAngTolerance)),pPhi-(2.*kPi),&pPhi);
    
    //*******************************
    
    Float_t tolAngMin = fSPhi + halfAngTolerance;
    Float_t tolAngMax = ePhi - halfAngTolerance;
    
    // Phi boundaries  : Do not check if it has no phi boundary!
    if(!unplaced.IsFullPhiSphere()) //Later May be done using MaskedAssign
    {
    completelyinside &= (pPhi <= tolAngMax) && (pPhi >= tolAngMin);
    
    tolAngMin = fSPhi - halfAngTolerance;
    tolAngMax = ePhi + halfAngTolerance;
    
    //std::cout<<std::setprecision(20)<<tolAngMin<<" : "<<tolAngMax<<" : "<<pPhi<<std::endl;
    completelyoutside |= (pPhi < tolAngMin) || (pPhi > tolAngMax);
    if( IsFull(completelyoutside) )return;
    }
    //Phi Check for GenericKernel Over
         
    // Theta bondaries
    //Float_t pTheta = localPoint.Theta();
    Float_t pTheta = ATan2(Sqrt(localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y()), localPoint.z()); //This needs to be implemented in Vector3D.h as Theta() function
    Float_t fSTheta(unplaced.GetStartThetaAngle());
    Float_t fDTheta(unplaced.GetDeltaThetaAngle());
    Float_t eTheta = fSTheta + fDTheta;
    
    tolAngMin = fSTheta + halfAngTolerance;
    tolAngMax = eTheta - halfAngTolerance;
    
    if(!unplaced.IsFullThetaSphere())
    {
        completelyinside &= (pTheta <= tolAngMax) && (pTheta >= tolAngMin);
       
        tolAngMin = fSTheta - halfAngTolerance;
        tolAngMax = eTheta + halfAngTolerance;
    
        //std::cout<<std::setprecision(20)<<tolAngMin<<" : "<<tolAngMax<<" : "<<pPhi<<std::endl;
        completelyoutside |= (pTheta < tolAngMin) || (pTheta > tolAngMax);
        if( IsFull(completelyoutside) )return;
        
    }
    
    return;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::Inside(UnplacedSphere const &unplaced,
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
void SphereImplementation<transCodeT, rotCodeT>::InsideKernel(UnplacedSphere const &unplaced,
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
VECGEOM_INLINE
void SphereImplementation<transCodeT, rotCodeT>::SafetyToIn(UnplacedSphere const &unplaced,
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
void SphereImplementation<transCodeT, rotCodeT>::SafetyToInKernel(UnplacedSphere const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety){

    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v      Bool_t;

    // Float_t safe=Backend::kZero;
    Float_t zero=Backend::kZero; 

    Vector3D<Float_t> localPoint;
    localPoint=point;

    //General Precalcs
    Float_t rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2);
    Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rho = std::sqrt(rho2);
    
    //Distance to r shells
    Float_t fRmin (unplaced.GetInnerRadius());
    Float_t safeRMin = unplaced.GetInnerRadius() - rad;
    Float_t safeRMax = rad - unplaced.GetOuterRadius();
    MaskedAssign(((fRmin > zero)&& (safeRMin > safeRMax)),safeRMin,&safety);
    MaskedAssign(((fRmin > zero)&& (!(safeRMin > safeRMax))),safeRMax,&safety);
    MaskedAssign((!(fRmin > zero)),safeRMax,&safety);
    //Distance to r shells over
    
    //Some Precalc
    Float_t fSPhi(unplaced.GetStartPhiAngle());
    Float_t fDPhi(unplaced.GetDeltaPhiAngle());
    Float_t hDPhi(unplaced.GetHDPhi()); // = (0.5 * fDPhi);
    Float_t cPhi(unplaced.GetCPhi()); //  =  fSPhi + hDPhi;
    Float_t ePhi(unplaced.GetEPhi()); //  = fSPhi + fDPhi;
    Float_t sinCPhi(unplaced.GetSinCPhi()); //   = std::sin(cPhi);
    Float_t cosCPhi(unplaced.GetCosCPhi()); //   = std::cos(cPhi);
    Float_t sinSPhi(unplaced.GetSinSPhi()); // = std::sin(fSPhi);
    Float_t cosSPhi(unplaced.GetCosSPhi()); // = std::cos(fSPhi);
    Float_t sinEPhi(unplaced.GetSinEPhi()); // = std::sin(ePhi);
    Float_t cosEPhi(unplaced.GetCosEPhi()); // = std::cos(ePhi);
    Float_t safePhi = zero;
    
    Float_t mone(-1.);
    
    Float_t cosPsi = (localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / rho; 
    //
    // Distance to phi extent
    //
    if(!unplaced.IsFullPhiSphere() && (rho>zero))
    {
        Float_t test1=((localPoint.x() * sinSPhi - localPoint.y() * cosSPhi));
        MaskedAssign((test1<0),mone*test1,&test1); //Facing the issue with abs function of Vc, Its actually giving the absolute value of floor function
        
        Float_t test2=((localPoint.x() * sinEPhi - localPoint.y() * cosEPhi));
        MaskedAssign((test2<0),mone*test2,&test2);
        
        MaskedAssign(((cosPsi < cos(hDPhi)) && ((localPoint.y() * cosCPhi - localPoint.x() * sinCPhi) <= 0)),test1,&safePhi);
        MaskedAssign(((cosPsi < cos(hDPhi)) && !((localPoint.y() * cosCPhi - localPoint.x() * sinCPhi) <= 0)),test2,&safePhi);
        MaskedAssign(((cosPsi < cos(hDPhi)) && (safePhi > safety)),safePhi,&safety);
                
    }
    //
    // Distance to Theta extent
    //
    Float_t KPI(kPi);
    Float_t rds = localPoint.Mag();
    Float_t piby2(kPi/2);
    Float_t pTheta = piby2 - asin(localPoint.z() / rds);
    
    MaskedAssign((pTheta<zero),pTheta+KPI,&pTheta);
    
    Float_t fSTheta(unplaced.GetStartThetaAngle());
    Float_t fDTheta(unplaced.GetDeltaThetaAngle());
    Float_t eTheta(unplaced.GetETheta()); //  = fSTheta + fDTheta;
    Float_t dTheta1 = fSTheta - pTheta;
    Float_t dTheta2 = pTheta-eTheta;
    Float_t sinDTheta1 = sin(dTheta1);
    Float_t sinDTheta2 = sin(dTheta2);
    Float_t safeTheta = zero;
    
    if(!unplaced.IsFullThetaSphere() && (rds!=zero))
    {
        MaskedAssign(((dTheta1 > dTheta2) && (dTheta1 >= zero)),(rds * sinDTheta1),&safeTheta);
        MaskedAssign(((dTheta1 > dTheta2) && (dTheta1 >= zero) && (safety <= safeTheta)),safeTheta,&safety);
        
        MaskedAssign((!(dTheta1 > dTheta2) && (dTheta2 >= zero)),(rds * sinDTheta2),&safeTheta);
        MaskedAssign((!(dTheta1 > dTheta2) && (dTheta2 >= zero) && (safety <= safeTheta)),safeTheta,&safety);
    }
    
    //Last line
    MaskedAssign( (safety < zero) , zero, &safety);
    
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SphereImplementation<transCodeT, rotCodeT>::SafetyToOut(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
    SafetyToOutKernel<Backend>(
    unplaced,
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
    //std::cout<<"Safety to OUT Kernel call"<<std::endl;
    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v      Bool_t;

    // Float_t safe=Backend::kZero;
    Float_t zero=Backend::kZero; 

    Vector3D<Float_t> localPoint;
    localPoint = point;
    
    //General Precalcs
    Float_t rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2);
    Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rho = Sqrt(rho2);
    
    //Distance to r shells
    //Float_t
    Precision fRmin = unplaced.GetInnerRadius();
    //Float_t 
    Precision fRmax = unplaced.GetOuterRadius();
    Float_t safeRMin = rad - fRmin;
    Float_t safeRMax = fRmax - rad ;
        
    //
    // Distance to r shells
    //
    MaskedAssign( ( (fRmin > zero) && (safeRMin < safeRMax) ),safeRMin,&safety);
    MaskedAssign( ( (fRmin > zero) && !(safeRMin < safeRMax) ),safeRMax,&safety);
    MaskedAssign( ( !(fRmin > zero) ),safeRMax,&safety);
    
    //
    // Distance to phi extent
    //
    
    //Some Precalc
    Float_t fSPhi (unplaced.GetStartPhiAngle());
    Float_t fDPhi (unplaced.GetDeltaPhiAngle());
    Float_t hDPhi (unplaced.GetHDPhi()); //(0.5 * fDPhi);
    Float_t cPhi  (unplaced.GetCPhi()); //fSPhi + hDPhi;
    Float_t ePhi  (unplaced.GetEPhi()); //fSPhi + fDPhi;
    Float_t sinCPhi (unplaced.GetSinCPhi()); //std::sin(cPhi);
    Float_t cosCPhi (unplaced.GetCosCPhi()); //std::cos(cPhi);
    Float_t sinSPhi (unplaced.GetSinSPhi()); // std::sin(fSPhi);
    Float_t cosSPhi (unplaced.GetCosSPhi()); //std::cos(fSPhi);
    Float_t sinEPhi (unplaced.GetSinEPhi()); //std::sin(ePhi);
    Float_t cosEPhi (unplaced.GetCosEPhi()); //std::cos(ePhi);
    Float_t safePhi = zero;
    
    Float_t mone(-1.);
    
    if(!unplaced.IsFullPhiSphere() && (rho>0))
    {
        Float_t test1 = (localPoint.y() * cosCPhi - localPoint.x() * sinCPhi);
        //Float_t test2 = (localPoint.x() * sinEPhi - localPoint.y() * cosEPhi);
        MaskedAssign( (test1<=zero) ,(mone*(localPoint.x() * sinSPhi - localPoint.y() * cosSPhi)), &safePhi);
        MaskedAssign( (!(test1<=zero)) ,(localPoint.x() * sinEPhi - localPoint.y() * cosEPhi), &safePhi);
        MaskedAssign( (safePhi < safety) ,safePhi , &safety);
        
    }
    
    //
    // Distance to Theta extent
    //
    Float_t KPI(kPi);
    Float_t rds = localPoint.Mag();
    Float_t piby2(kPi/2);
    Float_t pTheta = piby2 - asin(localPoint.z() / rds);
    
    MaskedAssign((pTheta<zero),pTheta+KPI,&pTheta);
    
    Float_t fSTheta(unplaced.GetStartThetaAngle());
    Float_t fDTheta(unplaced.GetDeltaThetaAngle());
    Float_t eTheta  = fSTheta + fDTheta;
    Float_t  dTheta1 =  pTheta - fSTheta;
    Float_t dTheta2 = eTheta - pTheta;
    
    Float_t sinDTheta1 = sin(dTheta1);
    Float_t sinDTheta2 = sin(dTheta2); 
    Float_t safeTheta = zero;
    
    if(!unplaced.IsFullThetaSphere() && (rds>0))
    {
        MaskedAssign(( (dTheta1 < dTheta2) ),(rds * sinDTheta1),&safeTheta);
        MaskedAssign((!(dTheta1 < dTheta2)),(rds * sinDTheta2),&safeTheta);
        MaskedAssign( (safety > safeTheta ),safeTheta,&safety);
    }
     
    //Last line
    MaskedAssign( (safety < zero) , zero, &safety);
}


 

/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToIn(UnplacedSphere const &unplaced,
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
void SphereImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){
    
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Float_t> localPoint;
    localPoint = point;

    Vector3D<Float_t> localDir;
    localDir = direction;

    //General Precalcs
    Float_t rad2 = localPoint.Mag2();
    Float_t rad = Sqrt(rad2);
    Float_t pDotV3d = localPoint.Dot(localDir);

    Float_t radius2 = unplaced.GetRadius() * unplaced.GetRadius();
    Float_t c = rad2 - radius2;
    Float_t d2 = pDotV3d * pDotV3d - c;

    Float_t pos_dot_dir_x = localPoint.x()*localDir.x();
    Float_t pos_dot_dir_y = localPoint.y()*localDir.y();
    Float_t pos_dot_dir_z = localPoint.z()*localDir.z();

    Bool_t done(false);
    distance = kInfinity;
    Float_t zero=Backend::kZero;

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

 */
 
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToOut(UnplacedSphere const &unplaced,
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
void SphereImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Float_t zero=Backend::kZero;
    Float_t PI(kPi);
    //distance = zero;
    
    Vector3D<Float_t> localPoint;
    localPoint = point;
    
    Vector3D<Float_t> localDir;
    localDir =  direction;
    
    //General Precalcs
    Float_t rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2); //r
    Float_t pDotV3d = localPoint.Dot(localDir); //rdotn
    Float_t rxy2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();

    Bool_t done(false);
    Bool_t tr(true),fal(false);
    distance = kInfinity;
    Float_t mone(-1.);
    Float_t snxt(kInfinity);
    Float_t one(1.);
    
    // Float_t KPI(kPi);
    // Float_t piby2(kPi/2);
    Float_t toler(1.E-10); 
    Float_t toler2(1.E+10); 
    Float_t b(0.), delta(0.), xnew(0.), ynew(0.), znew(0.), phi0(0.), ddp(0.);
    Float_t sn1(kInfinity),rn1(kInfinity);
    Vector3D<Float_t> newPt;
    Float_t fPhi1(unplaced.GetStartPhiAngle());
    Float_t fPhi2(unplaced.GetEPhi());
    Float_t fRmin(unplaced.GetInnerRadius());
    Float_t fRmax(unplaced.GetOuterRadius());
    Float_t dist(0.);
   
   //Inner Sphere
    done |= ((fRmin > zero) && (rad <= (fRmin+toler)) && (pDotV3d < zero) );
    MaskedAssign( ((fRmin > zero) &&  (rad <= (fRmin+toler)) && (pDotV3d < zero) ), kInfinity ,&distance);
    if( IsFull(done) )return ;
    DistanceToSphere<Backend>(unplaced,localPoint,localDir,fRmin,tr,tr,dist);
    MaskedAssign( ((fRmin > zero) &&  !(rad <= (fRmin+toler)) && (pDotV3d < zero) ), dist ,&rn1);
    
    Float_t sn2(kInfinity),rn2(kInfinity);
   //OuterSphere
    done |= ((rad >= (fRmax-toler)) && (pDotV3d >= zero));
    MaskedAssign(((rad >= (fRmax-toler)) && (pDotV3d >= zero)),kInfinity,&distance);
    if( IsFull(done) ) return;
    
    DistanceToSphere<Backend>(unplaced, localPoint, localDir, fRmax, fal, fal, rn2);
   
    Float_t sr = Min(rn1,rn2);
   
   // check theta conical surfaces
   
   Precision fTheta1 = unplaced.GetStartThetaAngle();
   Precision fTheta2 = unplaced.GetETheta();
   Precision theta2Plane=std::abs(fTheta2-kPi/2);
   Bool_t isFullPhiSphere(unplaced.IsFullPhiSphere());
   
   if(!unplaced.IsFullThetaSphere()){
       Precision theta1Plane=std::abs(fTheta1-kPi/2);
       if(theta1Plane < 1.e-10)
       MaskedAssign( ( (localPoint.z() * localDir.z()) <  zero) , mone*localPoint.z()/localDir.z(), &sn1);
       else{
               Bool_t cond1 = ( (fTheta1>zero) );
               Float_t r1(0.),r2(0.),z1(0.),z2(0.),dz(0.);
               Vector3D<Float_t> ptnew;
               Float_t si(unplaced.GetSinSTheta());
               Float_t ci(unplaced.GetCosSTheta());
               
               MaskedAssign( (cond1 && (ci > zero)) , fRmin*si ,&r1);
               MaskedAssign( (cond1 && (ci > zero)) , fRmin*ci ,&z1);
               MaskedAssign( (cond1 && (ci > zero)) , fRmax*si ,&r2);
               MaskedAssign( (cond1 && (ci > zero)) , fRmax*ci ,&z2);
               
               MaskedAssign( (cond1 && !(ci > zero)) , fRmax*si ,&r1);
               MaskedAssign( (cond1 && !(ci > zero)) , fRmax*ci ,&z1);
               MaskedAssign( (cond1 && !(ci > zero)) , fRmin*si ,&r2);
               MaskedAssign( (cond1 && !(ci > zero)) , fRmin*ci ,&z2);
               
               dz = 0.5*(z2-z1);
	       ptnew = localPoint;
               ptnew.z() = localPoint.z() - 0.5*(z1+z2);
               Float_t zinv = one/dz;
               Float_t rin = 0.5*(r1+r2+(r2-r1)*ptnew.z()*zinv);
               // Protection in case point is outside
               Float_t sigz(1.);
               MaskedAssign( (( localPoint.z() < zero )),mone*sigz,&sigz);
               
               Bool_t bigequ(false);
               bigequ = ( (sigz*ci>zero) && (sigz*rxy2 < sigz*rin*(rin+sigz*toler)));
	       Float_t ddotn(0.);
               Float_t ddotnTemp = ptnew.x()*localDir.x() + ptnew.y()*localDir.y() + 0.5*(r1-r2)*localDir.z()*zinv*Sqrt(rxy2);
               MaskedAssign((cond1 && bigequ),ddotnTemp,&ddotn);
               
               done |= (cond1 && (bigequ) && (sigz*ddotn <= zero) );
               MaskedAssign( ( cond1 && (bigequ) && (sigz*ddotn <= zero) ) , zero, &distance );  
               if( IsFull(done) ) return;
               
               Float_t deltav(0.),bv(0.);
               DistanceToCone<Backend>(unplaced, ptnew, localDir, dz, r1, r2, bv, deltav);
	       MaskedAssign((cond1 && !bigequ),deltav,&delta);
               MaskedAssign((cond1 && !bigequ),bv,&b);
	       
	       
               MaskedAssign((cond1 &&  (!bigequ) && (delta>zero)), (mone*b-delta), &snxt);
	    
               MaskedAssign((cond1 && (!bigequ) && (delta>zero)), (ptnew.z() + snxt*localDir.z()), &znew);
               
               Float_t temp=znew;
               MaskedAssign((temp<zero),mone*temp,&temp);
	    
	       Bool_t tempBigCond1 = (cond1 && (!bigequ) && (delta>zero) && (snxt>zero) && (temp < dz) && isFullPhiSphere);
	       MaskedAssign(tempBigCond1, snxt, &sn1);
               Bool_t bigcond3 = (cond1 && (!bigequ) && (delta>zero) && (snxt>zero) && (temp < dz) && (!isFullPhiSphere));
	       
               MaskedAssign(bigcond3, (ptnew.x() + snxt*localDir.x()), &xnew);
               MaskedAssign(bigcond3, (ptnew.y() + snxt*localDir.y()), &ynew);
               MaskedAssign(bigcond3, (ATan2(ynew,xnew)), &phi0);
               MaskedAssign(bigcond3, (phi0-fPhi1), &ddp);
               MaskedAssign((bigcond3 && (ddp<zero)), (ddp+2*kPi), &ddp);
               MaskedAssign((bigcond3 && (ddp<=(fPhi2-fPhi1))), snxt, &sn1);
               
               MaskedAssign((cond1 && (!bigequ) && (delta>zero) && (sn1>toler2)), mone*b+delta,&snxt);
               MaskedAssign((cond1 && (!bigequ) && (delta>zero) && (sn1>toler2)), (ptnew.z() + snxt*localDir.z()), &znew);
               
               temp=znew;
               MaskedAssign((temp<zero),mone*temp,&temp);
               
               MaskedAssign((cond1 && (!bigequ) && (delta>zero) && (sn1>toler2) && (snxt>zero) && (temp < dz) && isFullPhiSphere), snxt, &sn1);
               Bool_t bigcond4 = (cond1 && (!bigequ) && (delta>zero) && (sn1>toler2) && (snxt>zero) && (temp < dz) && (!isFullPhiSphere));
	       
               MaskedAssign(bigcond4, (ptnew.x() + snxt*localDir.x()), &xnew);
               MaskedAssign(bigcond4, (ptnew.y() + snxt*localDir.y()), &ynew);
               MaskedAssign(bigcond4, (ATan2(ynew,xnew)), &phi0);
               MaskedAssign(bigcond4, (phi0-fPhi1), &ddp);
               MaskedAssign((bigcond4 && (ddp<zero)), (ddp+2*kPi), &ddp);
               MaskedAssign((bigcond4 && (ddp<=(fPhi2-fPhi1))), snxt, &sn1);
             
       }
       
       Float_t Inf(kInfinity);
       snxt = Inf;
       Bool_t cond2 = ( (fTheta2<PI) );
       if(theta2Plane  <1.e-10)
           MaskedAssign( ((localPoint.z() * localDir.z()) <  zero) , mone*localPoint.z()/localDir.z(), &sn1);
       else{
               Float_t r1(0.),r2(0.),z1(0.),z2(0.),dz(0.);
               Vector3D<Float_t> ptnew;
               Float_t sei(unplaced.GetSinETheta());
               Float_t cei(unplaced.GetCosETheta());
	       
	       Float_t si=sei;
	       Float_t ci=cei;
               
               MaskedAssign( (cond2 && (ci > zero)) , fRmin*si ,&r1);
               MaskedAssign( (cond2 && (ci > zero)) , fRmin*ci ,&z1);
               MaskedAssign( (cond2 && (ci > zero)) , fRmax*si ,&r2);
               MaskedAssign( (cond2 && (ci > zero)) , fRmax*ci ,&z2);
               
               MaskedAssign( (cond2 && !(ci > zero)) , fRmax*si ,&r1);
               MaskedAssign( (cond2 && !(ci > zero)) , fRmax*ci ,&z1);
               MaskedAssign( (cond2 && !(ci > zero)) , fRmin*si ,&r2);
               MaskedAssign( (cond2 && !(ci > zero)) , fRmin*ci ,&z2);
               
               dz = 0.5*(z2-z1);
               	       
               ptnew = localPoint;
               ptnew.z() = localPoint.z() - 0.5*(z1+z2);
	       Float_t zinv = one/dz;
               Float_t rin = 0.5*(r1+r2+(r2-r1)*ptnew.z()*zinv);
               Float_t sigz=one;
               MaskedAssign( ( localPoint.z() < zero ),mone*sigz,&sigz);
               
               Bool_t bigequ = ( (sigz*ci>zero) && (sigz*rxy2 > sigz*rin*(rin-sigz*toler)));
	       Float_t ddotn=zero;
               Float_t ddotnTemp = ptnew.x()*localDir.x() + ptnew.y()*localDir.y() + 0.5*(r1-r2)*localDir.z()*zinv*Sqrt(rxy2);
               MaskedAssign((cond2 && bigequ),ddotnTemp,&ddotn);
               
               done |= ( (cond2 && bigequ) && (sigz*ddotn >= zero) );
               MaskedAssign( ( (cond2 && bigequ) && (sigz*ddotn >= zero) ) , zero, &distance );
               if( IsFull(done) ) return;
               
               Float_t deltav=zero;Float_t bv=zero;
               DistanceToCone<Backend>(unplaced, ptnew, localDir, dz, r1, r2, bv, deltav);
	       MaskedAssign((cond2 && !bigequ),deltav,&delta);
               MaskedAssign((cond2 && !bigequ),bv,&b);
               
	       MaskedAssign(( cond2 && (!bigequ) && (delta>zero)), (mone*b-delta), &snxt);
	       MaskedAssign(( cond2 && (!bigequ) && (delta>zero)), (ptnew.z() + snxt*localDir.z()), &znew);
               
               Float_t temp=znew;
               MaskedAssign((temp<zero),mone*temp,&temp);
	       Bool_t tempBigCond = ( cond2 && (!bigequ) && (delta>zero) && (snxt>zero) && (temp < dz) && isFullPhiSphere);
	       MaskedAssign(tempBigCond, snxt, &sn2);
               
	       Bool_t bigcond5 = ( cond2 && (!bigequ) && (delta>zero) && (snxt>zero) && (temp < dz) && (!isFullPhiSphere));
               MaskedAssign(bigcond5, (ptnew.x() + snxt*localDir.x()), &xnew);
               MaskedAssign(bigcond5, (ptnew.y() + snxt*localDir.y()), &ynew);
               MaskedAssign(bigcond5, (ATan2(ynew,xnew)), &phi0);
               MaskedAssign(bigcond5, (phi0-fPhi1), &ddp);
               MaskedAssign((bigcond5 && (ddp<zero)), (ddp+2*kPi), &ddp);
               MaskedAssign((bigcond5 && (ddp<=(fPhi2-fPhi1))), snxt, &sn2);
               MaskedAssign((cond2 &&  (!bigequ) && (delta>zero) && (sn2>toler)), mone*b+delta,&snxt);
	       MaskedAssign(( cond2 && (!bigequ) && (delta>zero) && (sn2>toler2)), (ptnew.z() + snxt*localDir.z()), &znew);
               
               temp=znew;
               MaskedAssign((temp<zero),mone*temp,&temp);
               MaskedAssign((cond2 &&  (!bigequ) && (delta>zero) && (sn2>toler2) && (snxt>zero) && (temp < dz) && isFullPhiSphere), snxt, &sn2);
               
	       Bool_t bigcond6 = ( cond2 && (!bigequ) && (delta>zero) && (sn2>toler2) && (snxt>zero) && (temp < dz) && (!isFullPhiSphere));
               MaskedAssign(bigcond6, (ptnew.x() + snxt*localDir.x()), &xnew);
               MaskedAssign(bigcond6, (ptnew.y() + snxt*localDir.y()), &ynew);
               MaskedAssign(bigcond6, (ATan2(ynew,xnew)), &phi0);
               MaskedAssign(bigcond6, (phi0-fPhi1), &ddp);
               MaskedAssign((bigcond6 && (ddp<zero)), (ddp+2*kPi), &ddp);
               MaskedAssign((bigcond6 && (ddp<=(fPhi2-fPhi1))), snxt, &sn2);
               
       }
       
       }
  
  //Theta cone distance calculation over
  
   Float_t st = Min(sn1,sn2);
  
   Float_t sp(kInfinity);
   if(!unplaced.IsFullPhiSphere()){
       //std::cout<<" -- NOT Full Phi Sphere --"<<std::endl;
       Float_t s1(unplaced.GetSinSPhi());
       Float_t c1(unplaced.GetCosSPhi());
       Float_t s2(unplaced.GetSinEPhi());
       Float_t c2(unplaced.GetCosEPhi());
       Float_t phim = 0.5*(fPhi1+fPhi2);
       Float_t sm = sin(phim);
       Float_t cm = cos(phim);
       DistanceToPhiMin<Backend>(unplaced,localPoint,localDir,s1,c1,s2,c2,sm,cm,tr,sp);
  }
   
   Float_t srt = Min(sr,st);
   distance = Min(srt,sp);
    
}


} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
