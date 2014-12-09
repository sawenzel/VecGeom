
/// @file SphereImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedSphere.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
namespace VECGEOM_NAMESPACE { 
 
 
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct SphereImplementation {
    

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
      typename Backend::bool_v &inside);
    
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedSphere const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedSphere const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside);

template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void GenericKernelForContainsAndInside(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedSphere const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      /*Vector3D<typename Backend::precision_v> const &n,  
      typename Backend::bool_v validNorm,    */
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      /*Vector3D<typename Backend::precision_v> const &n,  
      typename Backend::bool_v validNorm,    */
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);



template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedSphere const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

 template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);
 

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void ContainsKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside);
  

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void InsideKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside);

 
template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);



template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(UnplacedSphere const &unplaced,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);
  


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);
  
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Normal(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );
  
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ApproxSurfaceNormalKernel(
       UnplacedSphere const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal);
  
  
 
   
};


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
        
    // Distance to r shells
    
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
    
    // Distance to phi planes
    
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
   MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && (distSPhi < distEPhi) && (distSPhi < distMin)),distSPhi ,&distMin );
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && (distSPhi < distEPhi) && (distSPhi < prevDistMin)),kNSPhi ,&side ); //CULPRIT
    
    prevDistMin = distMin;
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && !(distSPhi < distEPhi) && (distEPhi < distMin)),distEPhi ,&distMin );
    MaskedAssign( ( !unplaced.IsFullPhiSphere() && (rho>zero) && !(distSPhi < distEPhi) && (distEPhi < prevDistMin)),kNEPhi ,&side );
    
    // Distance to theta planes
     
    temp = pTheta - fSTheta;
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero)) , temp*radius ,&distSTheta);
    
    temp = pTheta - fSTheta - fDTheta;
    MaskedAssign((temp<zero),(mone*temp),&temp);
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero)) , temp*radius ,&distETheta);
    
    prevDistMin = distMin;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && (distSTheta < distETheta) && (distSTheta < distMin)), distSTheta, &distMin); 
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && (distSTheta < distETheta) && (distSTheta < prevDistMin)), kNSTheta, &side); 
    
    prevDistMin = distMin;
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && !(distSTheta < distETheta) && (distETheta < distMin)), distETheta, &distMin); 
    MaskedAssign( ( !unplaced.IsFullThetaSphere() && (radius > zero) && !(distSTheta < distETheta) && (distETheta < prevDistMin)), kNETheta, &side);
    
    Bool_t done(false);
    done |= (side == kNRMin);
    MaskedAssign( (side == kNRMin), Vector3D<Float_t>(-localPoint.x() / radius, -localPoint.y() / radius, -localPoint.z() / radius),&norm);
   
    if( IsFull(done) )return ;
    
    done |= (side == kNRMax);
    MaskedAssign( (side == kNRMax),Vector3D<Float_t>(localPoint.x() / radius, localPoint.y() / radius, localPoint.z() / radius),&norm);
   if( IsFull(done) )return ;
    
    done |= (side == kNSPhi);
    MaskedAssign( (side == kNSPhi),Vector3D<Float_t>(sinSPhi, -cosSPhi, 0),&norm);
    
    if( IsFull(done) )return ;
    
    done |= (side == kNEPhi);
    MaskedAssign( (side == kNEPhi),Vector3D<Float_t>(-sinEPhi, cosEPhi, 0),&norm);
    if( IsFull(done) )return ;
    
    done |= (side == kNSTheta);
    MaskedAssign( (side == kNSTheta),Vector3D<Float_t>(-cosSTheta * std::cos(pPhi), -cosSTheta * std::sin(pPhi),sinSTheta),&norm);
    if( IsFull(done) )return ;
    
    done |= (side == kNETheta);
    MaskedAssign( (side == kNETheta),Vector3D<Float_t>(cosETheta * std::cos(pPhi), cosETheta * std::sin(pPhi), sinETheta),&norm);
    if( IsFull(done) )return ;
    
       
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
    
    
    Precision kAngTolerance = unplaced.GetAngTolerance();
    Precision halfAngTolerance = (0.5 * kAngTolerance);
    
    Float_t distSPhi(kInfinity),distSTheta(kInfinity);
    Float_t distEPhi(kInfinity),distETheta(kInfinity);
    Float_t distRMax(kInfinity); 
    Float_t distRMin(kInfinity);
    
    Vector3D<Float_t> nR, nPs, nPe, nTs, nTe, nZ(0., 0., 1.);
    Vector3D<Float_t> norm, sumnorm(0., 0., 0.);
    
    Bool_t fFullPhiSphere = unplaced.IsFullPhiSphere();
     Bool_t fFullThetaSphere = unplaced.IsFullThetaSphere();
    
    Float_t zero(0.);
    Float_t mone(-1.);
    
    Float_t temp=0;
    temp=radius - fRmax;
    MaskedAssign((temp<zero),mone*temp,&temp);
    distRMax = temp;
    
    distRMin = 0;
    temp = radius - fRmin;
    MaskedAssign((temp<zero),mone*temp,&temp);
    MaskedAssign( (fRmin > 0) , temp, &distRMin);
    
    MaskedAssign( ((rho>zero) && !(unplaced.IsFullSphere()) && (pPhi < fSPhi - halfAngTolerance)) , (pPhi+(2*kPi)), &pPhi);
    MaskedAssign( ((rho>zero) && !(unplaced.IsFullSphere()) && (pPhi > ePhi + halfAngTolerance)) , (pPhi-(2*kPi)), &pPhi);
    
    //Phi Stuff
    temp = pPhi - fSPhi;
    MaskedAssign((temp<zero),mone*temp,&temp);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) ), temp,&distSPhi);
    temp = pPhi - ePhi;
    MaskedAssign((temp<zero),mone*temp,&temp);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (rho>zero) ), temp,&distEPhi);
    
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (!fRmin) ), zero ,&distSPhi);
    MaskedAssign( (!unplaced.IsFullPhiSphere() && (!fRmin) ), zero ,&distEPhi);
    MaskedAssign( (!unplaced.IsFullPhiSphere()), Vector3D<Float_t>(sinSPhi, -cosSPhi, 0),&nPs );
    MaskedAssign( (!unplaced.IsFullPhiSphere()), Vector3D<Float_t>(-sinEPhi, cosEPhi, 0),&nPe );
    
    //Theta Stuff
    temp = pTheta - fSTheta;
    MaskedAssign((temp<zero),mone*temp,&temp);
     MaskedAssign( ( !unplaced.IsFullThetaSphere() && (rho>zero) ) , temp , &distSTheta ) ;
    
    temp = pTheta - eTheta;
    MaskedAssign((temp<zero),mone*temp,&temp);
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
  GenericKernelForContainsAndInside<Backend, false >(unplaced,
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
    typedef typename Backend::bool_v      Bool_t;	
    
    
    Float_t isfuPhiSph(unplaced.IsFullPhiSphere());
   
    Precision fRmin = unplaced.GetInnerRadius();
    Precision fRminTolerance = unplaced.GetFRminTolerance();
    Precision kAngTolerance = unplaced.GetAngTolerance();
    Precision halfAngTolerance = (0.5 * kAngTolerance);
    Precision fRmax = unplaced.GetOuterRadius();
        
    Float_t rad2 = localPoint.Mag2();
    //Float_t tolRMin = fRmin + (0.5 * fRminTolerance); 
    Float_t tolRMin(fRmin + (0.5 * fRminTolerance)); 
    Float_t tolRMin2 = tolRMin * tolRMin;
    //Float_t tolRMax = fRmax - (0.5 * fRminTolerance); 
    Float_t tolRMax(fRmax - (0.5 * fRminTolerance)); 
    Float_t tolRMax2 = tolRMax * tolRMax;
     
    
    completelyoutside = rad2 > MakeSPlusTolerantSquare<ForInside>( unplaced.GetOuterRadius(), unplaced.GetOuterRadius() * unplaced.GetOuterRadius() );//rmax
    if (ForInside)
    {
      completelyinside = rad2 < MakeSMinusTolerantSquare<ForInside>( unplaced.GetOuterRadius(), unplaced.GetOuterRadius() * unplaced.GetOuterRadius() );
    }
    if (Backend::early_returns) {
      if ( IsFull(completelyoutside) ) {
        return;
      }
    }
    
    completelyoutside |= rad2 < MakeSMinusTolerantSquare<ForInside>( unplaced.GetInnerRadius(), unplaced.GetInnerRadius() * unplaced.GetInnerRadius() );//rmin
    if (ForInside)
    {
      completelyinside &= rad2 > MakeSPlusTolerantSquare<ForInside>( unplaced.GetInnerRadius(), unplaced.GetInnerRadius() * unplaced.GetInnerRadius() );
    }

    // NOT YET NEEDED WHEN NOT PHI TREATMENT
    if (Backend::early_returns) {
        if ( IsFull(completelyoutside) ) {
            return;
          }
    }
    //Radial Check for GenericKernel Over
    
    Float_t tolAngMin(0.);
    Float_t tolAngMax(0.);        
    // Phi boundaries  : Do not check if it has no phi boundary!
    if(!unplaced.IsFullPhiSphere()) 
    {
    
     Bool_t completelyoutsidephi;
     Bool_t completelyinsidephi;
     unplaced.GetWedge().GenericKernelForContainsAndInside<Backend,ForInside>( localPoint,
     completelyinsidephi, completelyoutsidephi );
     completelyoutside |= completelyoutsidephi;
      if(ForInside)
            completelyinside &= completelyinsidephi;
        
    }
    //Phi Check for GenericKernel Over
         
    // Theta bondaries
   
    
    if(!unplaced.IsFullThetaSphere())
    {
    
             
     Bool_t completelyoutsidetheta;
     Bool_t completelyinsidetheta;
     unplaced.GetThetaCone().GenericKernelForContainsAndInside<Backend,ForInside>( localPoint,
     completelyinsidetheta, completelyoutsidetheta );
     completelyoutside |= completelyoutsidetheta;
     if(ForInside)
           completelyinside &= completelyinsidetheta;
      
        
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
    typedef typename Backend::bool_v      Bool_t;

    Float_t safe=Backend::kZero;
    Float_t zero=Backend::kZero; 

    Vector3D<Float_t> localPoint;
    localPoint=point;

    //General Precalcs
    Float_t rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2);
    Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rho = std::sqrt(rho2);
    
    //Distance to r shells
    Precision fRmin = unplaced.GetInnerRadius();
    Float_t fRminV(fRmin);
    Precision fRmax = unplaced.GetOuterRadius();
    Float_t fRmaxV(fRmax);
    Float_t safeRMin(0.);
    Float_t safeRMax(0.);
    
    if(fRmin)
    {
       safeRMin = fRminV - rad;
       safeRMax = rad - fRmaxV;
       CondAssign((safeRMin > safeRMax),safeRMin,safeRMax,&safety);
    }
    else
    {
        safety = rad - fRmaxV;
    }
    
    //Distance to r shells over
    
   
   
    if(!unplaced.IsFullPhiSphere())
    {
        
        Float_t safetyPhi = unplaced.GetWedge().SafetyToIn<Backend>(localPoint);
       safety = Max(safetyPhi,safety);
       
    }
    
    
     if(!unplaced.IsFullThetaSphere())
    {
         Float_t safeTheta = unplaced.GetThetaCone().SafetyToIn<Backend>(localPoint);
         safety = Max(safeTheta,safety);
    }
    
    
    
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
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Float_t zero=Backend::kZero; 

    Vector3D<Float_t> localPoint=point;
    Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rad=localPoint.Mag();
    Float_t rho = Sqrt(rho2);
    
    //Distance to r shells
    
    Precision fRmin = unplaced.GetInnerRadius();
    Float_t fRminV(fRmin);
    Precision fRmax = unplaced.GetOuterRadius();
    Float_t fRmaxV(fRmax);
   
    
    if(fRmin)
    {
        Float_t safeRMin=(rad - fRminV);
        Float_t safeRMax=(fRmaxV - rad); 
        CondAssign( ( (safeRMin < safeRMax) ),safeRMin,safeRMax,&safety);
    }
    else
    {
        safety = (fRmaxV - rad); 
    }
    
    // Distance to phi extent
    
    Float_t safePhi = zero;
    
    Float_t mone(-1.);
    
    if(!unplaced.IsFullPhiSphere() )
    {
        
        Float_t safetyPhi = unplaced.GetWedge().SafetyToOut<Backend>(localPoint);
       safety = Min(safetyPhi,safety);
    }
    
    // Distance to Theta extent
    
    Float_t safeTheta(0.);
    
    if(!unplaced.IsFullThetaSphere() )
    {  
        safeTheta = unplaced.GetThetaCone().SafetyToOut<Backend>(localPoint);
        safety = Min(safeTheta,safety);
        
    }
   
    MaskedAssign( ((safety < zero) || (safety < kTolerance)), zero, &safety);
}



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

    distance = kInfinity;
    Float_t rho2, rad2, pDotV2d, pDotV3d, pTheta;
    Float_t tolSTheta(0.), tolETheta(0.);
    
    
    //----------------------------------------------------------
    
    Bool_t done(false);
    Bool_t tr(true),fal(false);
    
    Float_t mone(-1.);
    Float_t one(1.);
    Float_t snxt(kInfinity);
    
    Precision fSTheta = unplaced.GetStartThetaAngle();
    Precision fETheta = unplaced.GetETheta();
    //Precision fSPhi = unplaced.GetStartPhiAngle();
    //Precision fEPhi = unplaced.GetEPhi();
    //Precision fabsTanSTheta = unplaced.GetFabsTanSTheta();
    //Precision fabsTanETheta = unplaced.GetFabsTanETheta();
    
    Float_t sinSPhi (unplaced.GetSinSPhi()); 
    Float_t cosSPhi (unplaced.GetCosSPhi()); 
    Float_t sinEPhi (unplaced.GetSinEPhi());
    Float_t cosEPhi (unplaced.GetCosEPhi()); 
    Float_t tanSTheta2(unplaced.GetTanSTheta2());
    Float_t tanETheta2(unplaced.GetTanETheta2());
    
    Float_t fSThetaV(fSTheta);
    Float_t fEThetaV(fETheta);
    
    Float_t zero=Backend::kZero;
    Float_t pi(kPi);
    Float_t piby2(kPi/2);
    //Float_t toler(1.E-10); 
    Float_t toler(kTolerance);
    Float_t toler2(1.E+10); 
    //Float_t halfCarTolerance(0.5 * 1e-9);
    //Float_t halfCarTolerance(0.5 * 1e-10);
    Float_t halfCarTolerance(kHalfTolerance);
    Precision kAngTolerance = unplaced.GetAngTolerance();
    Float_t halfAngTolerance(0.5 * kAngTolerance);
    //Float_t halfTolerance(kHalfTolerance);
    Float_t halfTolerance(0.5*unplaced.GetMKTolerance());
    Float_t fRminTolerance(unplaced.GetFRminTolerance());
    Float_t halfRminTolerance = 0.5 * fRminTolerance;
    Float_t fRmax(unplaced.GetOuterRadius()); Float_t fRmin(unplaced.GetInnerRadius()); 
    Float_t rMaxPlus = fRmax + halfTolerance;
    Float_t rMinMinus(0.); 
    MaskedAssign((fRmin > zero), (fRmin - halfTolerance), &rMinMinus);
    //----------------------------------------------------------------

    const Float_t dRmax = 100.*fRmax;
    
    Float_t tolORMin2(0.);
    MaskedAssign((fRmin > halfRminTolerance),((fRmin - halfRminTolerance) * (fRmin - halfRminTolerance)) ,&tolORMin2);
                           
    const Float_t tolIRMin2 =  (fRmin + halfRminTolerance) * (fRmin + halfRminTolerance);
    const Float_t tolORMax2 = (fRmax + halfTolerance) * (fRmax + halfTolerance);
    const Float_t tolIRMax2 = (fRmax - halfTolerance) * (fRmax - halfTolerance);
    
    Float_t cosCPhi(unplaced.GetCosCPhi());
    Float_t sinCPhi(unplaced.GetSinCPhi());
    Float_t cosHDPhiOT (unplaced.GetCosHDPhiOT());
    Float_t cosHDPhiIT (unplaced.GetCosHDPhiIT());

    Bool_t fFullPhiSphereV(unplaced.IsFullPhiSphere());
    Bool_t fFullThetaSphereV(unplaced.IsFullThetaSphere());
    
  // Intersection point
  Float_t xi(0.), yi(0.), zi(0.), rhoi(0.), rhoi2(0.), radi2(0.), iTheta(0.);

  // Phi intersection
  Float_t Comp;

  // Phi precalcs
  //
  Float_t Dist(0.), cosPsi(0.);

  // Theta precalcs
  Float_t dist2STheta, dist2ETheta;
  Float_t t1(0.), t2(0.), b(0.), c(0.), d2(0.), d(0.), sd(kInfinity);

  // General Precalcs
  rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
   rad2 = rho2 + localPoint.z() * localPoint.z();
   pTheta = ATan2(Sqrt(rho2), localPoint.z());

   pDotV2d = localPoint.x() * localDir.x() + localPoint.y() * localDir.y();
   pDotV3d = pDotV2d + localPoint.z() * localDir.z(); //localPoint.Dot(localDir);

   Bool_t cond(false);
  
  if (!unplaced.IsFullThetaSphere())
  {
    tolSTheta = fSThetaV - halfAngTolerance;
    tolETheta = fEThetaV + halfAngTolerance;
  }
  
  c = rad2 - fRmax * fRmax;
  
  
  MaskedAssign((c > kTolerance * fRmax),(pDotV3d * pDotV3d - c),&d2);
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) ) ,(mone*pDotV3d - Sqrt(d2)),&sd);
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) ) ,(localPoint.x() + sd * localDir.x()) ,&xi);
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) ) ,(localPoint.y() + sd * localDir.y()) ,&yi);
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) ) ,Sqrt(xi * xi + yi * yi),&rhoi);
  
  
  if(!unplaced.IsFullPhiSphere())
  {
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) && ((rhoi!=zero)) ),
          ((xi * cosCPhi + yi * sinCPhi) / rhoi),&cosPsi);
  
  
  if(!unplaced.IsFullThetaSphere())
  {
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) && ((rhoi!=zero)) &&
          (cosPsi >= cosHDPhiOT)  ),(localPoint.z() + sd * localDir.z()),&zi);
  
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) && ( (rhoi!=zero)) &&
          (cosPsi >= cosHDPhiOT)   ),(ATan2(rhoi,zi)),&iTheta);
  
  
  cond = ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) && ((rhoi!=zero)) &&
          (cosPsi >= cosHDPhiOT) && ((iTheta >= tolSTheta) && (iTheta <= tolETheta)) );
  MaskedAssign(cond ,sd,&snxt);
  done |= cond;
    
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
    
  }
  else
  {
  
  cond = ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) && ((rhoi!=zero)) &&
          (cosPsi >= cosHDPhiOT) );
  MaskedAssign( cond,sd,&snxt);
  done |= cond;
  
  if(IsFull(done))
  {   distance = snxt;
      return;
  }
  
  }
  }
  
  else
  {
   if(!unplaced.IsFullThetaSphere())
   {
  
  
  
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) ),(localPoint.z() + sd * localDir.z()),&zi);
  
  MaskedAssign( ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) ),(ATan2(rhoi, zi)),&iTheta);
  
  cond = ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) &&
           ((iTheta >= tolSTheta) && (iTheta <= tolETheta)) );
  MaskedAssign(cond ,sd,&snxt);
  done |= cond;
  
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
  }
   else
   {
  
  cond = ( (c > kTolerance * fRmax) && (d2 >= zero) && (sd >= zero) );
  MaskedAssign( cond ,sd,&snxt);
  done |= cond;
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
   }
  }
  
  cond = ( (c > kTolerance * fRmax) && !(d2 >= zero));
  MaskedAssign(cond,kInfinity,&snxt);
  done |= cond;
  
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
  MaskedAssign(( !(c > kTolerance * fRmax)),(pDotV3d * pDotV3d - c),&d2);
  
  Bool_t nCond(false);
  nCond =  (!(c > kTolerance * fRmax) && (rad2 > tolIRMax2)&& ((d2 >= kTolerance * fRmax) && (pDotV3d < zero)));
 
  if(!unplaced.IsFullPhiSphere())
  {
  MaskedAssign(( !(c > kTolerance * fRmax)  && nCond),
         ((localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / Sqrt(rho2)) ,&cosPsi);
  
  if(!unplaced.IsFullThetaSphere())
  {
  cond = ( !(c > kTolerance * fRmax)  && nCond && (cosPsi >= cosHDPhiIT) && 
   ((pTheta >= tolSTheta + kAngTolerance) && (pTheta <= tolETheta - kAngTolerance)));
  MaskedAssign((cond && !done),zero,&snxt);
  done |= cond;
  
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
  }
  else
  { 
  cond = ( !(c > kTolerance * fRmax) && nCond && (cosPsi >= cosHDPhiIT) );
  MaskedAssign((cond  && !done),zero,&snxt);
  done |= cond;
  
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
   
  }
  }
  else
  {
      if(!unplaced.IsFullThetaSphere())
      {
        cond = ( !(c > kTolerance * fRmax) && nCond &&
          ((pTheta >= tolSTheta + kAngTolerance) && (pTheta <= tolETheta - kAngTolerance))  );
        MaskedAssign((cond  && !done),zero,&snxt);
        done|= cond;
  
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
      }
      else
      {
        cond = ( !(c > kTolerance * fRmax) && nCond );
        
       
        
  MaskedAssign((cond  && !done),zero,&snxt);
  done |= cond;
  
  if(IsFull(done))
  {
      distance = snxt;
      return;
  }
  
      }
  }
  
  Precision checkRmin=unplaced.GetInnerRadius();
  
  if(checkRmin)
  {
       c = rad2 - fRmin * fRmin;
      d2 = pDotV3d * pDotV3d - c;
      if(!unplaced.IsFullPhiSphere())
      {
      MaskedAssign(( ((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero)))  ),((localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / Sqrt(rho2)),&cosPsi);
      if(!unplaced.IsFullThetaSphere())
      {
	 cond = ( ((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero)))  && (cosPsi >= cosHDPhiIT) && 
              ((pTheta >= tolSTheta + kAngTolerance) && (pTheta <= tolETheta - kAngTolerance)) );
      MaskedAssign((cond && !done),zero,&snxt);
      done |= cond;
     
      if(IsFull(done))
      {
      distance = snxt;
      return;
      }
      
      }
      else
      {
      cond = ( ((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (cosPsi >= cosHDPhiIT) );
      MaskedAssign((cond && !done),zero,&snxt);
      done |= cond;
     
      
      if(IsFull(done))
      {
      distance = snxt;
      return;
      }
      
      }
      }
      else
      {  
          if(!unplaced.IsFullThetaSphere())
          {
	    
      cond = ( ((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero)))  &&
              ((pTheta >= tolSTheta + kAngTolerance) && (pTheta <= tolETheta - kAngTolerance)) );
      MaskedAssign((cond && !done),zero,&snxt);
      done |= cond;
     
      if(IsFull(done))
      {
      distance = snxt;
      return;
      }
      
          }
          else
          {
       
      cond = ( ((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) );
      MaskedAssign((cond && !done),zero,&snxt);
      done |= cond;
     
      if(IsFull(done))
      {
      distance = snxt;
      return;
      }
      
          }
      }
      
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) ),(mone*pDotV3d + Sqrt(d2)),&sd);
      
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) ),(localPoint.x() + sd * localDir.x()),&xi);
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) ),(localPoint.y() + sd * localDir.y()),&yi);
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) ),(Sqrt(xi * xi + yi * yi)),&rhoi);
      
      if(!unplaced.IsFullPhiSphere())
      {
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ( (rhoi != zero)) ),
              ((xi * cosCPhi + yi * sinCPhi) / rhoi),&cosPsi);
      
      if(!unplaced.IsFullThetaSphere())
      {
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero)) &&
              (cosPsi >= cosHDPhiOT) ),(localPoint.z() + sd * localDir.z()),&zi);
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero)) &&
              (cosPsi >= cosHDPhiOT) ),(ATan2(rhoi, zi)),&iTheta);
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero)) &&
              (cosPsi >= cosHDPhiOT) && ((iTheta >= tolSTheta) && (iTheta <= tolETheta))  && !done),sd,&snxt);
      }
      else
      {
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero)) &&
              (cosPsi >= cosHDPhiOT) && !done ),sd,&snxt);
      }
      
      }
      else
      {
	  if(!unplaced.IsFullThetaSphere())
          {
	     MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero))  ),
              (localPoint.z() + sd * localDir.z()),&zi);
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero))  ),
              (ATan2(rhoi, zi)),&iTheta);
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero)) &&
               ((iTheta >= tolSTheta) && (iTheta <= tolETheta))&& !done ),sd,&snxt);
      
      Bool_t debCond = ( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) && ((rhoi != zero)) &&
               ((iTheta >= tolSTheta) && (iTheta <= tolETheta)) );
      
          }
          else
          {
      
      MaskedAssign(( !((c > mone*halfRminTolerance) && (rad2 < tolIRMin2) && 
              ((d2 < fRmin * toler) || (pDotV3d >= zero))) && (d2 >= 0) && (sd >= halfRminTolerance) /*&& !( (rhoi != zero))*/ && !done),sd,&snxt);
      
      
          }
      
  }
  }
  
  
  if(!unplaced.IsFullPhiSphere())
  {
      
      Comp = localDir.x() * sinSPhi - localDir.y() * cosSPhi;
      MaskedAssign((Comp < zero),(localPoint.y() * cosSPhi - localPoint.x() * sinSPhi),&Dist);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance)),(Dist / Comp),&sd);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > zero)),(localPoint.x() + sd * localDir.x()),&xi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > zero)),(localPoint.y() + sd * localDir.y()),&yi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > zero)),(localPoint.z() + sd * localDir.z()),&zi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > zero)),(xi * xi + yi * yi),&rhoi2);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > zero)),(rhoi2 + zi * zi ),&radi2);
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > zero)),zero,&sd);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > zero)),(localPoint.x()),&xi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > zero)),(localPoint.y()),&yi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > zero)),(localPoint.z()),&zi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > zero)),rho2,&rhoi2);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > zero)),rad2,&radi2);
      
      if(!unplaced.IsFullThetaSphere())
      {
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && ((radi2 <= tolORMax2)
              && (radi2 >= tolORMin2)
              && ((yi * cosCPhi - xi * sinCPhi) <= zero)) ),(ATan2(Sqrt(rhoi2),zi)),&iTheta);
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && ((radi2 <= tolORMax2)
              && (radi2 >= tolORMin2)
              && ((yi * cosCPhi - xi * sinCPhi) <= zero))  && ((iTheta >= tolSTheta) && (iTheta <= tolETheta)) &&
               ((yi * cosCPhi - xi * sinCPhi) <= zero) && !done),sd,&snxt);
      }
      else
      {
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && ((radi2 <= tolORMax2)
              && (radi2 >= tolORMin2)
              && ((yi * cosCPhi - xi * sinCPhi) <= zero)) && !done ),sd,&snxt);
     
      }
      
      // Second phi surface ('E'nding phi)
     

      Comp = mone*(localDir.x() * sinEPhi - localDir.y() * cosEPhi);
      MaskedAssign((Comp < zero),(mone*(localPoint.y() * cosEPhi - localPoint.x() * sinEPhi)),&Dist);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) ),(Dist / Comp),&sd);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > 0) ),(localPoint.x() + sd * localDir.x()),&xi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > 0) ),(localPoint.y() + sd * localDir.y()),&yi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > 0) ),(localPoint.z() + sd * localDir.z()),&zi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > 0) ),(xi * xi + yi * yi),&rhoi2);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && (sd > 0) ),(rhoi2 + zi * zi),&radi2 );
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > 0) ),zero,&sd);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > 0) ),(localPoint.x()),&xi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > 0) ),(localPoint.y()),&yi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > 0) ),(localPoint.z()),&zi);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > 0) ),rho2,&rhoi2);
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && !(sd > 0) ),rad2,&radi2);
      
      if(!unplaced.IsFullThetaSphere())
      {
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && ((radi2 <= tolORMax2)
              && (radi2 >= tolORMin2)
              && ((yi * cosCPhi - xi * sinCPhi) >= 0)) ),(ATan2(Sqrt(rhoi2), zi)),&iTheta);
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && ((radi2 <= tolORMax2)
              && (radi2 >= tolORMin2)
              && ((yi * cosCPhi - xi * sinCPhi) >= zero))  && ((iTheta >= tolSTheta) && (iTheta <= tolETheta))
              && ((yi * cosCPhi - xi * sinCPhi) >= zero) && !done),sd,&snxt);
      
      }
      else
      {
      
      MaskedAssign(((Comp < zero) && (Dist < halfCarTolerance) && (sd < snxt) && ((radi2 <= tolORMax2)
              && (radi2 >= tolORMin2)
              && ((yi * cosCPhi - xi * sinCPhi) >= zero))&& !done ),sd,&snxt);
     
      }
      

      
  }
  
  
  
  Bool_t thetaCond(false);
Float_t INF(kInfinity);  
Bool_t set(false);
  if(!unplaced.IsFullThetaSphere())
  {
    
    if (fSTheta)
    {
      dist2STheta = rho2 - localPoint.z() * localPoint.z() * tanSTheta2;
    }
    else
    {
      dist2STheta = INF;
    }
    
    if (fETheta < kPi)
    {
      dist2ETheta = rho2 - localPoint.z() * localPoint.z() * tanETheta2;
    }
    else
    {
      dist2ETheta = INF;
    }
    
    set = (pTheta < tolSTheta);
    MaskedAssign(( (pTheta < tolSTheta) ),(one - localDir.z() * localDir.z() * (one + tanSTheta2)),&t1);
    MaskedAssign(( (pTheta < tolSTheta) ),(pDotV2d - localPoint.z() * localDir.z() * tanSTheta2),&t2);
    thetaCond = ( (pTheta < tolSTheta) && (t1 != zero) );
    MaskedAssign(thetaCond,(t2 / t1),&b);
    MaskedAssign(thetaCond,(dist2STheta / t1),&c);
    MaskedAssign(thetaCond,(b * b - c),&d2);
    
    thetaCond = ( (pTheta < tolSTheta) && (t1 != zero) && (d2 >= zero));
    MaskedAssign(thetaCond,Sqrt(d2),&d);
    MaskedAssign(thetaCond,(mone*b - d),&sd);
    MaskedAssign(thetaCond,(localPoint.z() + sd * localDir.z()),&zi);
    MaskedAssign(( thetaCond && ((sd < zero) || (zi * (fSTheta - piby2) > zero))),(mone*b+d),&sd);
    
    thetaCond = ( (pTheta < tolSTheta) && (t1 != zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) );
    MaskedAssign(thetaCond,(localPoint.x() + sd * localDir.x()),&xi);
    MaskedAssign(thetaCond,(localPoint.y() + sd * localDir.y()),&yi);
    MaskedAssign(thetaCond,(localPoint.z() + sd * localDir.z()),&zi);
    MaskedAssign(thetaCond,(xi * xi + yi * yi),&rhoi2);
    MaskedAssign(thetaCond,(rhoi2 + zi * zi),&radi2);
    
    if(!unplaced.IsFullPhiSphere())
    {
      
    MaskedAssign(( (pTheta < tolSTheta) && (t1 != zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
            ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSTheta - piby2) <= zero)) && ((rhoi2!=zero)) ),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
    
    MaskedAssign(( (pTheta < tolSTheta) && (t1 != zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
            ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSTheta - piby2) <= zero)) && ((rhoi2!=zero)) && (cosPsi >= cosHDPhiOT)&& !done ),sd,&snxt);
    
    }
    else
    {
    MaskedAssign(( (pTheta < tolSTheta) && (t1 != zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
            ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSTheta - piby2) <= zero)) && ((rhoi2!=zero))&& !done),sd,&snxt);
    }
         
    thetaCond = ( (pTheta < tolSTheta) && (fEThetaV < kPi));
    MaskedAssign(thetaCond,(one - localDir.z() * localDir.z() * (one + tanETheta2)),&t1);
    MaskedAssign(thetaCond,(pDotV2d - localPoint.z() * localDir.z() * tanETheta2),&t2);
    
    thetaCond = ( (pTheta < tolSTheta) && (fEThetaV < kPi) && (t1!=zero));
    MaskedAssign(thetaCond,(t2 / t1),&b);        
    MaskedAssign(thetaCond,(dist2ETheta / t1),&c);
    MaskedAssign(thetaCond,(b * b - c),&d2);
    
    thetaCond = ( (pTheta < tolSTheta) && (fEThetaV < kPi) && (t1!=zero) && (d2 >= 0) );
    MaskedAssign(thetaCond,Sqrt(d2),&d);
    MaskedAssign(thetaCond,(mone*b + d),&sd);
    
    thetaCond= ( (pTheta < tolSTheta) && (fEThetaV < kPi) && (t1!=zero) && (d2 >= 0) && ((sd >= zero) && (sd < snxt)) );
    MaskedAssign(thetaCond,(localPoint.x() + sd * localDir.x()),&xi);
    MaskedAssign(thetaCond,(localPoint.y() + sd * localDir.y()),&yi);
    MaskedAssign(thetaCond,(localPoint.z() + sd * localDir.z()),&zi);
    MaskedAssign(thetaCond,(xi * xi + yi * yi),&rhoi2);
    MaskedAssign(thetaCond,(rhoi2 + zi * zi),&radi2);
    
    if(!unplaced.IsFullPhiSphere())
    {
      
    MaskedAssign(( (pTheta < tolSTheta) && (fEThetaV < kPi) && (t1!=zero) && (d2 >= 0) && ((sd >= zero) && (sd < snxt)) &&
            ((radi2 <= tolORMax2)
                  && (radi2 >= tolORMin2)
                  && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero)) ),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
    
    MaskedAssign(( (pTheta < tolSTheta) && (fEThetaV < kPi) && (t1!=zero) && (d2 >= 0) && ((sd >= zero) && (sd < snxt)) &&
            ((radi2 <= tolORMax2)
                  && (radi2 >= tolORMin2)
                  && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && (cosPsi >= cosHDPhiOT) && !done),sd ,&snxt);  
    }
    else
    {
      
        MaskedAssign(( (pTheta < tolSTheta) && (fEThetaV < kPi) && (t1!=zero) && (d2 >= 0) && ((sd >= zero) && (sd < snxt)) &&
            ((radi2 <= tolORMax2)
                  && (radi2 >= tolORMin2)
                  && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero))&& !done),sd,&snxt);
        
    }
    
    
    MaskedAssign(((pTheta > tolETheta) && (!set)),(one - localDir.z() * localDir.z() * (one + tanETheta2)),&t1);
    MaskedAssign(((pTheta > tolETheta) && (!set)),(pDotV2d - localPoint.z() * localDir.z() * tanETheta2),&t2);
    //set |= (pTheta > tolETheta);
    
    thetaCond = ( (pTheta > tolETheta) && (!set) && (t1!=zero) );
    MaskedAssign( thetaCond,(t2 / t1),&b);
    MaskedAssign(thetaCond ,(dist2ETheta / t1),&c);
    MaskedAssign(thetaCond ,(b * b - c),&d2);
   
    thetaCond = ( (pTheta > tolETheta) && (!set) && (t1!=zero) && (d2 >= zero) );
    MaskedAssign(thetaCond ,Sqrt(d2),&d);
    MaskedAssign(thetaCond ,(mone*b - d),&sd);
    MaskedAssign(thetaCond ,(localPoint.z() + sd * localDir.z()),&zi);
    
    MaskedAssign(( (pTheta > tolETheta) && (!set) && (t1!=zero) && (d2 >= zero) && ((sd < zero) || (zi * (fEThetaV - piby2) > zero)) ) ,(mone*b+d),&sd);
   
    thetaCond = ( (pTheta > tolETheta) && (!set) && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)));
    MaskedAssign(thetaCond,(localPoint.x() + sd * localDir.x()),&xi);
    MaskedAssign(thetaCond,(localPoint.y() + sd * localDir.y()),&yi);
    MaskedAssign(thetaCond,(localPoint.z() + sd * localDir.z()),&zi);
    MaskedAssign(thetaCond,(xi * xi + yi * yi),&rhoi2);
    MaskedAssign(thetaCond,(rhoi2 + zi * zi),&radi2);
    
    if(!unplaced.IsFullPhiSphere())
    {
      
    MaskedAssign(( (pTheta > tolETheta) && (!set) && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
            ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero)) ),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
    
    MaskedAssign(( (pTheta > tolETheta) && (!set) && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
            ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && (cosPsi >= cosHDPhiOT)&& !done ),sd,&snxt);
   }
    else
    {
      
   MaskedAssign(( (pTheta > tolETheta) && (!set) && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
            ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero))&& !done),sd,&snxt);
  
    }
  
  
   
   MaskedAssign(( (pTheta > tolETheta) && (!set) && (fSThetaV!=zero)),(one - localDir.z() * localDir.z() * (one + tanSTheta2)),&t1);
   MaskedAssign(( (pTheta > tolETheta) && (!set) && (fSThetaV!=zero)),(pDotV2d - localPoint.z() * localDir.z() * tanSTheta2),&t2);
   MaskedAssign(( (pTheta > tolETheta) && (!set) && (fSThetaV!=zero) && (t1!=zero) ),(t2 / t1),&b);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) ),(dist2STheta / t1),&c);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) ),(b * b - c),&d2);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) ),Sqrt(d2),&d);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) ),(mone*b+d),&sd);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) ),(localPoint.x() + sd * localDir.x()),&xi);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) ),(localPoint.y() + sd * localDir.y()),&yi);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) ),(localPoint.z() + sd * localDir.z()),&zi);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) ),(xi * xi + yi * yi),&rhoi2);
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) ),(rhoi2 + zi * zi),&radi2);
   
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) &&
           ((radi2 <= tolORMax2)
                  && (radi2 >= tolORMin2)
                  && (zi * (fSThetaV - piby2) <= zero)) && ( (rhoi2!=zero)) ),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
   
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) &&
           ((radi2 <= tolORMax2)
                  && (radi2 >= tolORMin2)
                  && (zi * (fSThetaV - piby2) <= zero)) && ( (rhoi2!=zero)) && (cosPsi >= cosHDPhiOT)&& !done),sd,&snxt);
   }
   else
   {
     
   MaskedAssign(( (pTheta > tolETheta) && (!set)  && (fSThetaV!=zero) && (t1!=zero) && (d2 >= zero) && ((sd >= 0) && (sd < snxt)) &&
           ((radi2 <= tolORMax2)
                  && (radi2 >= tolORMin2)
                  && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero))&& !done),sd,&snxt);
   
   }
   
   set |= (pTheta > tolETheta);
   
   Float_t tempDist(0.);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set) ),(pDotV2d - localPoint.z() * localDir.z() * tanSTheta2),&t2);
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && 
           ((t2 >= zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta < piby2)
          || (t2 < zero  && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta > piby2)
          || (localDir.z() < zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSThetaV==piby2)) && ((rho2!=zero)) ),
           ((localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / Sqrt(rho2)),&cosPsi);
   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && 
           ((t2 >= zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta < piby2)
          || (t2 < zero  && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta > piby2)
          || (localDir.z() < zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSThetaV==piby2)) && ((rho2!=zero)) && (cosPsi >= cosHDPhiIT) && !done),zero,&snxt);//&tempDist);
  
   done |= ((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && 
           ((t2 >= zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta < piby2)
          || (t2 < zero  && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta > piby2)
          || (localDir.z() < zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSThetaV==piby2)) && ( (rho2!=zero)) && (cosPsi >= cosHDPhiIT) && !done);
   
   if(IsFull(done))
   {
       distance = snxt;
       return;
   }
   
   }
   else
   {
     
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && 
           ((t2 >= zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta < piby2)
          || (t2 < zero  && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta > piby2)
          || (localDir.z() < zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSThetaV==piby2)) && ((rho2!=zero))&& !done),zero,&snxt);//&tempDist);
   
   done |= ((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && 
           ((t2 >= zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta < piby2)
          || (t2 < zero  && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSTheta > piby2)
          || (localDir.z() < zero && tolIRMin2 < rad2 && rad2 < tolIRMax2 && fSThetaV==piby2)) && ((rho2!=zero)) && !done);
   
   
   if(IsFull(done))
   {
       distance = snxt;
       return;
   }
   
   }
   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set) ),(one - localDir.z() * localDir.z() * (one + tanSTheta2)),&t1);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero)),(t2 / t1),&b);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero)),(dist2STheta / t1),&c);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero)),(b * b - c),&d2);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero)),Sqrt(d2),&d);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero)),(mone*b+d),&sd);
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) ),(localPoint.x() + sd * localDir.x()),&xi);   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) ),(localPoint.y() + sd * localDir.y()),&yi);   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) ),(localPoint.z() + sd * localDir.z()),&zi);   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) ),(xi * xi + yi * yi),&rhoi2);   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) ),(rhoi2 + zi * zi),&radi2);   
   
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero))),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
   
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && (cosPsi >= cosHDPhiOT)&& !done),sd,&snxt); 
  }
   else
   {
     
   MaskedAssign(((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance) && (!set)  && (t1!=zero) && (d2 >= zero) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fSThetaV < piby2)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero))  && !done),sd,&snxt);
  
   }
   
   set |= ((pTheta < tolSTheta + kAngTolerance) && (fSTheta > halfAngTolerance));
   
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)  ),(pDotV2d - localPoint.z() * localDir.z() * tanETheta2),&t2);
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   &&
           (((t2 < zero) && (fEThetaV < piby2)
           && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((t2 >= zero) && (fEThetaV >  piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((localDir.z() > zero) && (fEThetaV == piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))) && ((rho2!=zero)) ),((localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / Sqrt(rho2)),&cosPsi);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   &&
           (((t2 < zero) && (fEThetaV < piby2)
           && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((t2 >= zero) && (fEThetaV >  piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((localDir.z() > zero) && (fEThetaV == piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))) && ((rho2!=zero)) && (cosPsi >= cosHDPhiIT)&& !done),zero,&snxt);//tempDist);
   
   
   done |= ((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   &&
           (((t2 < zero) && (fEThetaV < piby2)
           && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((t2 >= zero) && (fEThetaV >  piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((localDir.z() > zero) && (fEThetaV == piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))) && ((rho2!=zero)) && (cosPsi >= cosHDPhiIT) && !done);
   
   if(IsFull(done))
   {
       distance=snxt;
       return;
   }
   
   }
   else
   {
      
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   &&
           (((t2 < zero) && (fEThetaV < piby2)
           && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((t2 >= zero) && (fEThetaV >  piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((localDir.z() > zero) && (fEThetaV == piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))) && ((rho2!=zero))&& !done),zero,&snxt);
              
           
  done |=  ((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   &&
           (((t2 < zero) && (fEThetaV < piby2)
           && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((t2 >= zero) && (fEThetaV >  piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))
          || ((localDir.z() > zero) && (fEThetaV == piby2)
              && (tolIRMin2 < rad2) && (rad2 < tolIRMax2))) && ((rho2!=zero)) && !done);        
  
   if(IsFull(done))
   {
       distance=snxt;
       return;
   }
   
   }
   
   
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)  ),(one - localDir.z() * localDir.z() * (one + tanETheta2)),&t1);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero)),(t2 / t1),&b);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero)),(dist2ETheta / t1),&c);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero)),(b * b - c),&d2);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero) && (d2 >= 0)),Sqrt(d2),&d);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero) && (d2 >= 0)),(mone*b + d),&sd);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) ),( localPoint.x() + sd * localDir.x()),&xi);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) ),( localPoint.y() + sd * localDir.y()),&yi);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) ),( localPoint.z() + sd * localDir.z()),&zi);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)   && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) ),(xi * xi + yi * yi),&rhoi2);
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)  && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) ),(rhoi2 + zi * zi),&radi2);
   
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)  && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero))),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
   
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)  && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ( (rhoi2!=zero)) && (cosPsi >= cosHDPhiOT) && !done),sd,&snxt);
  }
   else
   {
     
   MaskedAssign(((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance) && (!set)  && (t1!=zero) && (d2 >= 0) && 
           ((sd >= halfCarTolerance) && (sd < snxt) && (fEThetaV > piby2)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ( (rhoi2!=zero))&& !done),sd,&snxt);
   }
   
   set |= ((pTheta > tolETheta - kAngTolerance) && (fEThetaV < kPi - kAngTolerance));
   
   Bool_t newCond = ( (pTheta > tolSTheta + kAngTolerance ) && (pTheta < tolETheta - kAngTolerance )  && (!set) );
   MaskedAssign(newCond,(one - localDir.z() * localDir.z() * (one + tanSTheta2)),&t1);
   MaskedAssign(newCond,(pDotV2d - localPoint.z() * localDir.z() * tanSTheta2),&t2);
   MaskedAssign((newCond && (t1!=zero)),(t2 / t1),&b);
   MaskedAssign((newCond && (t1!=zero)),(dist2STheta / t1),&c);
   MaskedAssign((newCond && (t1!=zero)),(b * b - c),&d);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) ),Sqrt(d2),&d);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) ),(mone*b + d),&sd);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(localPoint.x() + sd * localDir.x()),&xi);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(localPoint.y() + sd * localDir.y()),&yi);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(localPoint.z() + sd * localDir.z()),&zi);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(xi * xi + yi * yi),&rhoi2);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(rhoi2 + zi * zi),&radi2);
   
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
           ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero)) ),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
   
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
           ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && (cosPsi >= cosHDPhiOT) && !done),sd,&snxt);
   }
   else
   {
   
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && 
           ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fSThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && !done),sd,&snxt);
      
   }
   
   MaskedAssign((newCond),(one - localDir.z() * localDir.z() * (one + tanETheta2)),&t1);
   MaskedAssign((newCond),(pDotV2d - localPoint.z() * localDir.z() * tanETheta2),&t2);
   MaskedAssign((newCond && (t1!=zero)),(t2 / t1),&b);
   MaskedAssign((newCond && (t1!=zero)),(dist2ETheta / t1),&c);
   MaskedAssign((newCond && (t1!=zero)),(b * b - c),&d2);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) ),Sqrt(d2),&d);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) ),(mone*b+d),&sd);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(localPoint.x() + sd * localDir.x()),&xi);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(localPoint.y() + sd * localDir.y()),&yi);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(localPoint.z() + sd * localDir.z()),&zi);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(xi * xi + yi * yi),&rhoi2);
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) ),(rhoi2 + zi * zi),&radi2);
   
   if(!unplaced.IsFullPhiSphere())
    {
      
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero))  ),((xi * cosCPhi + yi * sinCPhi) / Sqrt(rhoi2)),&cosPsi);
   
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && (cosPsi >= cosHDPhiOT)&& !done ),sd,&snxt);    
  }
   else
   {
     
   MaskedAssign((newCond && (t1!=zero) && (d2 >= zero) && ((sd >= zero) && (sd < snxt)) && ((radi2 <= tolORMax2)
                && (radi2 >= tolORMin2)
                && (zi * (fEThetaV - piby2) <= zero)) && ((rhoi2!=zero)) && !done),sd,&snxt);
        
   }
  }
  
  distance = snxt;
   
}


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
    /*n,
    validNorm,*/
    stepMax,
    distance
  );
} 

//V3
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      /*Vector3D<typename Backend::precision_v> const &n,  
      typename Backend::bool_v validNorm,  */
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance){

    bool verbose=false;
    
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    
    Vector3D<Float_t> localPoint;
    localPoint = point;
    
    Vector3D<Float_t> localDir;
    localDir =  direction;
    
    
    Bool_t done(false);
    Bool_t tr(true),fal(false);
    distance = kInfinity;
    Float_t mone(-1.);
    Float_t one(1.);
    Float_t snxt(kInfinity);
    
    Precision fSTheta = unplaced.GetStartThetaAngle();
    Precision fETheta = unplaced.GetETheta();
    
    Precision fabsTanSTheta = unplaced.GetFabsTanSTheta();
    Precision fabsTanETheta = unplaced.GetFabsTanETheta();
    
    Float_t zero=Backend::kZero;
    Float_t pi(kPi);
    Float_t piby2(kPi/2);
    Float_t toler(1.E-10); 
    Float_t toler2(1.E+10); 
    Float_t halfCarTolerance(0.5 * 1e-9);
    Precision kAngTolerance = unplaced.GetAngTolerance();
    Float_t halfAngTolerance(0.5 * kAngTolerance);
    Float_t halfTolerance(0.5*unplaced.GetMKTolerance());
    Float_t mkTolerance(unplaced.GetMKTolerance());
    Float_t fRminTolerance(unplaced.GetFRminTolerance());
    Float_t halfRminTolerance = 0.5 * fRminTolerance;
    Float_t fRmax(unplaced.GetOuterRadius()); Float_t fRmin(unplaced.GetInnerRadius()); 
    Float_t rMaxPlus = fRmax + halfTolerance;
    Float_t rMinMinus(0.); 
    MaskedAssign((fRmin > zero), (fRmin - halfTolerance), &rMinMinus);
    
    
    //Variables for Phi intersection
    Float_t pDistS(0.), compS(0.), pDistE(0.), compE(0.), sphi2(0.), vphi(0.);
    
    Float_t rho2(0.),rad2(0.),pDotV2d(0.), pDotV3d(0.);
    
    Vector3D<Float_t> newPt ; //Intersection Point
    Float_t xi(0.),yi(0.),zi(0.);
   
    Float_t dist2STheta(0.),dist2ETheta(0.), distTheta(0.), d2(0.), sd(0.);
    Float_t t1(0.),t2(0.),b(0.),c(0.),d(0.);
    
    
    Float_t stheta(kInfinity),sphi(kInfinity);
    
    //General Precalcs
    rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2); //r
    pDotV3d = localPoint.Dot(localDir); //rdotn
    rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    pDotV2d = localPoint.x() * localDir.x() + localPoint.y() * localDir.y();

    Bool_t cond1 = ((rad2 <= rMaxPlus * rMaxPlus) && (rad2 >= rMinMinus * rMinMinus));
    MaskedAssign((cond1),(rad2 - fRmax * fRmax),&c);
    MaskedAssign(( (cond1) && (c < mkTolerance*fRmax)),(pDotV3d * pDotV3d - c),&d2);
    MaskedAssign(( (cond1) && (c < mkTolerance*fRmax) && 
            ((c > mone * mkTolerance * fRmax) && ((pDotV3d >= zero) || (d2 < zero))) ),zero,&snxt);
    
   
    done |= ( (cond1) && (c < mkTolerance*fRmax) && 
            ((c > mone * mkTolerance * fRmax) && ((pDotV3d >= zero) || (d2 < zero))) );
    
    if(IsFull(done))
    {
        distance = snxt;
        return;
    }
    
    MaskedAssign(( (cond1) && (c < mkTolerance*fRmax) && 
            !((c > mone * mkTolerance * fRmax) && ((pDotV3d >= zero) || (d2 < zero))) && !done),mone*pDotV3d+Sqrt(d2),&snxt);
    
    MaskedAssign(( (cond1) && (fRmin > zero)),(rad2 - fRmin*fRmin),&c);
    MaskedAssign(( (cond1) && (fRmin > zero)),(pDotV3d * pDotV3d - c),&d2);
    MaskedAssign(( (cond1) && (fRmin > zero) && (c > mone * fRminTolerance * fRmin) &&
            ((c < fRminTolerance * fRmin) && (d2 >= fRminTolerance * fRmin) && (pDotV3d < zero)) && !done) ,zero,&snxt);
    
    
    done |= ( (cond1) && (fRmin > zero) && (c > mone * fRminTolerance * fRmin) &&
            ((c < fRminTolerance * fRmin) && (d2 >= fRminTolerance * fRmin) && (pDotV3d < zero)) && !done);
    
    if(IsFull(done))
    {
        distance = snxt;
        return;
    }
    
    MaskedAssign(( (cond1) && (fRmin > zero) && (c > mone * fRminTolerance * fRmin) &&
            !((c < fRminTolerance * fRmin) && (d2 >= fRminTolerance * fRmin) && (pDotV3d < zero)) && (d2 >= zero) ),(mone*pDotV3d - Sqrt(d2)),&sd);
    
    MaskedAssign(( (cond1) && (fRmin > zero) && (c > mone * fRminTolerance * fRmin) &&
            !((c < fRminTolerance * fRmin) && (d2 >= fRminTolerance * fRmin) && (pDotV3d < zero)) && (d2 >= zero) && (sd >= zero) && !done),sd,&snxt);
    
    
    Float_t temp(0.),temp2(0.),temp3(0.);
    //Theta Segment Intersection
       
    Precision tanSTheta2 = unplaced.GetTanSTheta2();
    Precision tanSTheta = unplaced.GetTanSTheta();
    Float_t fSThetaV(fSTheta);
    
    Precision tanETheta2 = unplaced.GetTanETheta2();
    Precision tanETheta = unplaced.GetTanETheta();
    Float_t fEThetaV(fETheta);
    
    done = fal;
    
    if(!unplaced.IsFullThetaSphere())
    {  
        if(fSTheta)
        {
            if(fabsTanSTheta > 5. / kAngTolerance)
            {
                MaskedAssign(( (localDir.z() > zero) ) ,mone*localPoint.z()/localDir.z(),&stheta);
            }
            else
            {
                t1 = one - localDir.z() * localDir.z() * (1 + tanSTheta2);
                t2 = pDotV2d - localPoint.z() * localDir.z() * tanSTheta2;
                dist2STheta = rho2 - localPoint.z() * localPoint.z() * tanSTheta2;
                distTheta = Sqrt(rho2) - localPoint.z() * tanSTheta;
                
                MaskedAssign(((Abs(t1) < halfTolerance) && (localDir.z() >  zero) && (Abs(distTheta) < halfTolerance) && ((fSThetaV < piby2)&& (localPoint.z() > zero)) && !done ),zero,&stheta);
                done |= ( (Abs(t1) < halfTolerance) && (localDir.z() >  zero) && (Abs(distTheta) < halfTolerance) && ((fSThetaV < piby2)&& (localPoint.z() > zero)) && !done);
                if(IsFull(done))
                {
                    distance = stheta;
                    return ;
                }
                
                MaskedAssign(( (Abs(t1) < halfTolerance) && (localDir.z() >  zero) && (Abs(distTheta) < halfTolerance) && ((fSThetaV > piby2) && (localPoint.z() <= zero)) && !done),zero,&stheta);
                done |= ( (Abs(t1) < halfTolerance) && (localDir.z() >  zero) && (Abs(distTheta) < halfTolerance) && ((fSThetaV > piby2) && (localPoint.z() <= zero)) && !done);
                if(IsFull(done))
                {
                    distance = stheta;
                    return ;
                }
                
                MaskedAssign(( (Abs(t1) < halfTolerance) && (localDir.z() >  zero) && !(Abs(distTheta) < halfTolerance) && !done),(mone*0.5 * dist2STheta / t2),&stheta);
                
                MaskedAssign(( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fSThetaV > piby2) && (t2 >= zero)) && !done) , zero,&stheta);
                done |= ( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fSThetaV > piby2) && (t2 >= zero)) && !done);
                if(IsFull(done))
                {
                    distance = stheta;
                    return ;
                }
                
                MaskedAssign(( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fSThetaV < piby2) && (t2 < zero) && (localPoint.z() >= zero)) && !done) , zero,&stheta);
                done |= ( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fSThetaV < piby2) && (t2 < zero) && (localPoint.z() >= zero))&& !done);
                if(IsFull(done))
                {
                    distance = stheta;
                    return ;
                }
                
                MaskedAssign(!(Abs(t1) < halfTolerance),(t2/t1),&b);
                MaskedAssign(!(Abs(t1) < halfTolerance),(dist2STheta/t1),&c);
                MaskedAssign(!(Abs(t1) < halfTolerance),(b*b - c),&d2);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) ),Sqrt(d2),&d);
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && (fSThetaV > piby2) ),(mone*b-d),&sd);
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && (fSThetaV > piby2) && 
                        (((Abs(sd) < halfTolerance) && (t2 < zero))
                  || (sd < zero) || ((sd > zero) && (localPoint.z() + sd * localDir.z() > zero)))),mone*b+d,&sd);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && (fSThetaV > piby2) && 
                       ((sd > halfTolerance) && (localPoint.z() + sd * localDir.z() <= zero)) && !done),sd,&stheta);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && !(fSThetaV > piby2)),mone*b-d,&sd);
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && !(fSThetaV > piby2) && 
                        (((Abs(sd) < halfTolerance) && (t2 >= zero))
                  || (sd < zero) || ((sd > zero) && (localPoint.z() + sd * localDir.z() < zero))) ),mone*b+d,&sd);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && !(fSThetaV > piby2) && 
                        ((sd > halfTolerance) && (localPoint.z() + sd * localDir.z() >= zero)) && !done ),sd,&stheta);
                
                
                
            }
        }
        
        if(fETheta < kPi)
        {
            if(fabsTanETheta > 5. / kAngTolerance)
            {
                //MaskedAssign(( (localDir.z() < zero) && (Abs(localPoint.z()) <= halfTolerance)  ) ,zero,&stheta);
                MaskedAssign(( (localDir.z() < zero) && (Abs(localPoint.z()) <= halfTolerance) && !done ) ,zero,&snxt);
                done|=( (localDir.z() < zero) && (Abs(localPoint.z()) <= halfTolerance) && !done );
                if(IsFull(done))
                {
                    distance = snxt;
                    return;
                }
                
                MaskedAssign(( (localDir.z() < zero) ),(mone*localPoint.z()/localDir.z()),&sd);
                
                MaskedAssign(( (localDir.z() < zero) && (sd < stheta) && !done),sd,&stheta);
                
            }
            else
            {
                t1 = one - localDir.z() * localDir.z() * (1 + tanETheta2);
                t2 = pDotV2d - localPoint.z() * localDir.z() * tanETheta2;
                dist2ETheta = rho2 - localPoint.z() * localPoint.z() * tanETheta2;
                distTheta = Sqrt(rho2) - localPoint.z() * tanETheta;
                
                MaskedAssign(((Abs(t1) < halfTolerance) && (localDir.z() <  zero) && (Abs(distTheta) < halfTolerance) &&
                        ((fEThetaV > piby2)&& (localPoint.z() < zero)) && !done),zero,&snxt);
                done |= ((Abs(t1) < halfTolerance) && (localDir.z() <  zero) && (Abs(distTheta) < halfTolerance) &&
                        ((fEThetaV > piby2)&& (localPoint.z() < zero)) && !done);
                if(IsFull(done))
                {
                    distance = snxt;
                    return ;
                }
                
                MaskedAssign(( (Abs(t1) < halfTolerance) && (localDir.z() <  zero) && (Abs(distTheta) < halfTolerance) && 
                        ((fEThetaV < piby2) && (localPoint.z() >= zero))&& !done ),zero,&snxt);
                done |= ( (Abs(t1) < halfTolerance) && (localDir.z() <  zero) && (Abs(distTheta) < halfTolerance) && ((fEThetaV < piby2) && (localPoint.z() >= zero)) && !done);
                if(IsFull(done))
                {
                    distance = snxt;
                    return ;
                }
                
                //MaskedAssign(( (Abs(t1) < halfTolerance) && (localDir.z() <  zero) && !(Abs(distTheta) < halfTolerance) ),(mone*0.5 * dist2ETheta / t2),&sd);
                MaskedAssign(( (Abs(t1) < halfTolerance) && (localDir.z() <  zero) ),(mone*0.5 * dist2ETheta / t2),&sd);
                MaskedAssign(( (Abs(t1) < halfTolerance) && (localDir.z() <  zero) && (sd < stheta) && !done),sd,&stheta);
                
                MaskedAssign(( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV < piby2) && (t2 >= zero))&& !done) , zero,&stheta);
                done |= ( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV < piby2) && (t2 >= zero))&& !done);
                if(IsFull(done))
                {
                    distance = stheta;
                    return ;
                }
                
                //MaskedAssign(( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV > piby2)
                     //&& (t2 < zero) && (localPoint.z() <= zero)) ),zero,&stheta);
                MaskedAssign(( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV < piby2)
                     && (t2 >= zero) && (localPoint.z() <= zero)) && !done),zero,&snxt);
                
                done |= ( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV < piby2)
                     && (t2 >= zero) && (localPoint.z() <= zero)) && !done);
                if(IsFull(done))
                {
                    distance = snxt;
                    return ;
                }
                
                
                MaskedAssign(( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV > kPi / 2)
                     && (t2 < zero) && (localPoint.z() <= zero)) && !done),zero,&snxt);
                
                done |= ( !(Abs(t1) < halfTolerance) && (Abs(distTheta) < halfTolerance) && ((fEThetaV > kPi / 2)
                     && (t2 < zero) && (localPoint.z() <= zero))&& !done );
                
                if(IsFull(done))
                {
                    distance = snxt;
                    return ;
                }
                
                MaskedAssign(!(Abs(t1) < halfTolerance),(t2/t1),&b);
                MaskedAssign(!(Abs(t1) < halfTolerance),(dist2ETheta/t1),&c);
                MaskedAssign(!(Abs(t1) < halfTolerance),(b*b - c),&d2);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) ),Sqrt(d2),&d);
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && (fEThetaV < piby2) ),(mone*b-d),&sd);
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && (fEThetaV < piby2) && 
                        (((Abs(sd) < halfTolerance) && (t2 < zero))
                  || (sd < zero))),mone*b+d,&sd);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && (fEThetaV < piby2) &&
                       (sd > halfTolerance) && (sd < stheta) && !done),sd,&stheta);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && !(fEThetaV < piby2)),mone*b-d,&sd);
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && !(fEThetaV < piby2) && 
                        (((Abs(sd) < halfTolerance) && (t2 >= zero))
                  || (sd < zero) || ((sd > zero) && (localPoint.z() + sd * localDir.z() > zero))) ),mone*b+d,&sd);
                
                MaskedAssign((!(Abs(t1) < halfTolerance) && (d2>=zero) && !(fEThetaV < piby2) &&  
                        ((sd > halfTolerance) && (localPoint.z() + sd * localDir.z() <= zero)) && (sd < stheta) && !done),sd,&stheta);
                
                
                
            }
        }
        
        
    }
    
    //Added for Wedge otherwise not required
    distance = Min(stheta,snxt);
    //Theta Ends here
    
    // Phi Intersection
    Float_t INF(kInfinity);
    if (!unplaced.IsFullPhiSphere())
    {
      
      //Trying to use Wedge
      Float_t distPhi1;
      Float_t distPhi2;
      unplaced.GetWedge().DistanceToOut<Backend>(localPoint,localDir,distPhi1,distPhi2);
     
      Float_t distPhiMin = Min(distPhi1, distPhi2);
      distance = Min(distPhiMin,distance);
    
    }
 }


} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
