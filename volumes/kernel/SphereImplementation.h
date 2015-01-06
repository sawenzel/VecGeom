
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
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v  CheckPhiTheta(UnplacedSphere const &unplaced,
                                                Vector3D<typename Backend::precision_v> const localPoint, Vector3D<typename Backend::precision_v> const localDir,
                                                typename Backend::precision_v sd, /*typename Backend::precision_v &dist,*/ typename Backend::bool_v done );
  
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v  CheckSpecialTolerantCase(UnplacedSphere const &unplaced,
                                                Vector3D<typename Backend::precision_v> const localPoint, Vector3D<typename Backend::precision_v> const localDir,
                                                bool ifRmin, typename Backend::bool_v done );
  
  
  /*
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContainsDisk( UnplacedSphere const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {

 
    // TODO: do this generically WITH a generic contains/inside kernel
    // forget about sector for the moment
  
    ContainsKernel<Backend,false>(unplaced, point, inside);
}
   */
   
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
    Precision kAngTolerance = unplaced.GetAngTolerance()*10. ;
    Precision halfAngTolerance = (0.5 * kAngTolerance);
    Precision fRmax = unplaced.GetOuterRadius();
        
    Float_t rad2 = localPoint.Mag2();
    //Float_t tolRMin = fRmin + (0.5 * fRminTolerance); 
    
    //Float_t tolRMin(fRmin + (0.5 * fRminTolerance *10. )); 
    //Float_t tolRMin2 = tolRMin * tolRMin;
    
    Float_t tolRMin(fRmin + ( fRminTolerance *10. )); 
    Float_t tolRMin2 = tolRMin * tolRMin;
    
    //Float_t tolRMax = fRmax - (0.5 * fRminTolerance); 
    //Float_t tolRMax(fRmax - (0.5 * unplaced.GetMKTolerance() * 10. )); 
    //Float_t tolRMax2 = tolRMax * tolRMax;
    Float_t tolRMax(fRmax - ( unplaced.GetMKTolerance() * 10. )); 
    Float_t tolRMax2 = tolRMax * tolRMax;
    
    // Check radial surfaces
    //Radial check for GenericKernel Start
    
    completelyinside = (rad2 <= tolRMax*tolRMax) && (rad2 >= tolRMin*tolRMin);
    
    tolRMin = fRmin - (0.5 * fRminTolerance*10); 
    tolRMax = fRmax + (0.5 * unplaced.GetMKTolerance()*10); 
    
    completelyoutside = (rad2 <= tolRMin*tolRMin) || (rad2 >= tolRMax*tolRMax);
    if( IsFull(completelyoutside) )return;
    
    //Radial Check for GenericKernel Over
    
    Float_t tolAngMin(0.);
    Float_t tolAngMax(0.);        
    // Phi boundaries  : Do not check if it has no phi boundary!
    if(!unplaced.IsFullPhiSphere()) 
    {
       /*
     //   std::cout<<"Entered Full Phi Check"<<std::endl;
    Float_t pPhi = localPoint.Phi();
    Float_t fSPhi(unplaced.GetStartPhiAngle());
    Float_t fDPhi(unplaced.GetDeltaPhiAngle());
    Float_t ePhi = fSPhi+fDPhi;
    
    //*******************************
    //Very important 
    MaskedAssign((pPhi<(fSPhi - halfAngTolerance)),pPhi+(2.*kPi),&pPhi);
    MaskedAssign((pPhi>(ePhi + halfAngTolerance)),pPhi-(2.*kPi),&pPhi);
    
    //*******************************
    
    Float_t tolAngMin = fSPhi + halfAngTolerance;
    Float_t tolAngMax = ePhi - halfAngTolerance;
        
        
        
    completelyinside &= (pPhi <= tolAngMax) && (pPhi >= tolAngMin);
    
    tolAngMin = fSPhi - halfAngTolerance;
    tolAngMax = ePhi + halfAngTolerance;
    
    completelyoutside |= (pPhi < tolAngMin) || (pPhi > tolAngMax);
    std::cout<<"COMP INSIDE : "<<completelyinside<<"  :: COMP OUTSIDE : "<<completelyoutside<<std::endl;
    if( IsFull(completelyoutside) )return;
     
    */
     Bool_t completelyoutsidephi;
     Bool_t completelyinsidephi;
     unplaced.GetWedge().GenericKernelForContainsAndInside<Backend,ForInside>( localPoint,
     completelyinsidephi, completelyoutsidephi );
     completelyoutside |= completelyoutsidephi;
     if(ForInside)
            completelyinside &= completelyinsidephi;
           
    //   completelyoutside |= completelyoutsidephi;     
       //std::cout<<"COMP INSIDE : "<<completelyinside<<"  :: COMP OUTSIDE : "<<completelyoutside<<std::endl;
     //if( IsFull(completelyoutside) )return;
        
    }
    //Phi Check for GenericKernel Over
         
    // Theta bondaries
   
    
    if(!unplaced.IsFullThetaSphere())
    {
    
        /*
    Float_t pTheta = ATan2(Sqrt(localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y()), localPoint.z()); //This needs to be implemented in Vector3D.h as Theta() function
    Float_t fSTheta(unplaced.GetStartThetaAngle());
    Float_t fDTheta(unplaced.GetDeltaThetaAngle());
    Float_t eTheta = fSTheta + fDTheta;
    
    tolAngMin = fSTheta + halfAngTolerance;
    tolAngMax = eTheta - halfAngTolerance;
    
        
        completelyinside &= (pTheta <= tolAngMax) && (pTheta >= tolAngMin);
       
        tolAngMin = fSTheta - halfAngTolerance;
        tolAngMax = eTheta + halfAngTolerance;
    
        completelyoutside |= (pTheta < tolAngMin) || (pTheta > tolAngMax);
        if( IsFull(completelyoutside) )return;
     */
         
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

    Vector3D<Float_t> localPoint = point;
    
    //General Precalcs
    Float_t rad2    = localPoint.Mag2();
    Float_t rad = Sqrt(rad2);
    Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
    Float_t rho = Sqrt(rho2);
    
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
    
    //Some Precalc
    /*
    Float_t fSPhi(unplaced.GetStartPhiAngle());
    Float_t fDPhi(unplaced.GetDeltaPhiAngle());
    Float_t hDPhi(unplaced.GetHDPhi()); 
    Float_t cPhi(unplaced.GetCPhi()); 
    Float_t ePhi(unplaced.GetEPhi()); 
    Float_t sinCPhi(unplaced.GetSinCPhi()); 
    Float_t cosCPhi(unplaced.GetCosCPhi()); 
    Float_t sinSPhi(unplaced.GetSinSPhi()); 
    Float_t cosSPhi(unplaced.GetCosSPhi()); 
    Float_t sinEPhi(unplaced.GetSinEPhi()); 
    Float_t cosEPhi(unplaced.GetCosEPhi()); 
    Float_t safePhi = zero;
    Float_t mone(-1.);
    Float_t cosPsi(0.);
    */
    // Distance to phi extent
   
    if(!unplaced.IsFullPhiSphere())
    {
        /*
        MaskedAssign((rho > zero),( (localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / rho),&cosPsi);
        MaskedAssign(((rho > zero) && (cosPsi < cos(hDPhi)) && ((localPoint.y() * cosCPhi - localPoint.x() * sinCPhi) <= zero) ),
                (Abs(localPoint.x() * sinSPhi - localPoint.y() * cosSPhi)),&safePhi);
        MaskedAssign(((rho > zero) && (cosPsi < cos(hDPhi)) && !((localPoint.y() * cosCPhi - localPoint.x() * sinCPhi) <= zero) ),
                (Abs(localPoint.x() * sinEPhi - localPoint.y() * cosEPhi)),&safePhi);
        
        MaskedAssign(((rho > zero) && (cosPsi < cos(hDPhi)) && (safePhi > safety)),safePhi,&safety);
         
        */
        Float_t safetyPhi = unplaced.GetWedge().SafetyToIn<Backend>(localPoint);
        safety = Max(safetyPhi,safety);
       
    }
    
    // Distance to Theta extent
    /*
    Float_t KPI(kPi);
    Float_t rds = localPoint.Mag();
    Float_t piby2(kPi/2);
    
    Float_t fSTheta(unplaced.GetStartThetaAngle());
    Float_t fDTheta(unplaced.GetDeltaThetaAngle());
    Float_t eTheta(unplaced.GetETheta()); 
    Float_t dTheta1(0.);
    Float_t dTheta2(0.);
    Float_t safeTheta(0.);
    Float_t pTheta(0.);
    if(!unplaced.IsFullThetaSphere())
    {
        MaskedAssign((rad != zero),(piby2 - asin(localPoint.z() / rad)),&pTheta);
        MaskedAssign(((rad != zero) && (pTheta < zero)),(pTheta+KPI),&pTheta);
        MaskedAssign(((rad != zero) ),(fSTheta - pTheta),&dTheta1);
        MaskedAssign(((rad != zero) ),(pTheta - eTheta),&dTheta2);
        MaskedAssign(((rad != zero) && (dTheta1 > dTheta2) && (dTheta1 >= zero)),(rad * sin(dTheta1)),&safeTheta);
        MaskedAssign(((rad != zero) && (dTheta1 > dTheta2) && (dTheta1 >= zero) && (safety <= safeTheta) ),safeTheta,&safety);
        MaskedAssign(((rad != zero) && !(dTheta1 > dTheta2) && (dTheta2 >= zero)),(rad * sin(dTheta2)),&safeTheta);
        MaskedAssign(((rad != zero) && !(dTheta1 > dTheta2) && (dTheta2 >= zero) && (safety <= safeTheta)),safeTheta,&safety);
        
    }
   */
    
     if(!unplaced.IsFullThetaSphere())
    {
         //Float_t safeTheta = unplaced.GetThetaCone().SafetyToIn<Backend>(localPoint);
         //std::cout<<"SafeTheta : "<<safeTheta<<"  :: Safety : "<<safety<<std::endl;
         //safety = Max(safeTheta,safety);
         unplaced.GetThetaCone().SafetyToIn<Backend>(localPoint,safety);
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


//V2
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(UnplacedSphere const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety){
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

   // Float_t safe=Backend::kZero;
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
    
    //Float_t safeRMin(0.);
    //Float_t safeRMax(0.);
            
    
    // Distance to r shells
    
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
        /*
        Float_t sinCPhi (unplaced.GetSinCPhi()); //std::sin(cPhi);
        Float_t cosCPhi (unplaced.GetCosCPhi()); //std::cos(cPhi);
        Float_t sinSPhi (unplaced.GetSinSPhi()); // std::sin(fSPhi);
        Float_t cosSPhi (unplaced.GetCosSPhi()); //std::cos(fSPhi);
        Float_t sinEPhi (unplaced.GetSinEPhi()); //std::sin(ePhi);
        Float_t cosEPhi (unplaced.GetCosEPhi()); //std::cos(ePhi);
        Float_t safePhi = zero;
        //Float_t test1 = (localPoint.y() * cosCPhi - localPoint.x() * sinCPhi);
        MaskedAssign( (((localPoint.y() * cosCPhi - localPoint.x() * sinCPhi) <= zero) && (rho > zero)),(mone*(localPoint.x() * sinSPhi - localPoint.y() * cosSPhi)), &safePhi);
        MaskedAssign( (!((localPoint.y() * cosCPhi - localPoint.x() * sinCPhi) <= zero) && (rho > zero)) ,(localPoint.x() * sinEPhi - localPoint.y() * cosEPhi), &safePhi);
        MaskedAssign( ((safePhi < safety)) ,safePhi , &safety);
        */
        Float_t safetyPhi = unplaced.GetWedge().SafetyToOut<Backend>(localPoint);
       safety = Min(safetyPhi,safety);
    }
    
    // Distance to Theta extent
    
    Float_t safeTheta(0.);
    
    if(!unplaced.IsFullThetaSphere() )
    {  
        /*
        //Float_t KPI(kPi);
        //Float_t piby2(kPi/2);
        //Float_t fSTheta(unplaced.GetStartThetaAngle());
       // Float_t eTheta(unplaced.GetETheta());
        Float_t pTheta(0.);
        Float_t dTheta1(0.);
        Float_t dTheta2(0.);
    
        //MaskedAssign((rad > zero),(piby2 - asin(localPoint.z() / rad)),&pTheta);
        MaskedAssign((rad > zero),(kPi/2 - asin(localPoint.z() / rad)),&pTheta);
        //MaskedAssign( ((rad > zero) && (pTheta < zero) ),(pTheta+KPI),&pTheta);
        MaskedAssign( ((rad > zero) && (pTheta < zero) ),(pTheta+kPi),&pTheta);
        //MaskedAssign( ((rad > zero)),(pTheta - fSTheta),&dTheta1);
        MaskedAssign( ((rad > zero)),(pTheta - unplaced.GetStartThetaAngle()),&dTheta1);
        //MaskedAssign( ((rad > zero)),(eTheta - pTheta),&dTheta2);
        MaskedAssign( ((rad > zero)),(unplaced.GetETheta() - pTheta),&dTheta2);
        
        CondAssign((dTheta1 < dTheta2),(rad * sin(dTheta1)),(rad * sin(dTheta2)),&safeTheta);
        MaskedAssign( ((safeTheta < safety)) ,safeTheta , &safety);
        */
        safeTheta = unplaced.GetThetaCone().SafetyToOut<Backend>(localPoint);
        safety = Min(safeTheta,safety);
        
    }
    
    
    //MaskedAssign( (safety < zero) , zero, &safety);
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

/*
  //Original
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

    Vector3D<Float_t> localPoint = point;
    Vector3D<Float_t> localDir = direction;
    distance = kInfinity;
    
    //----------------------------------------------------------
    
    Bool_t done(false);
    
    //Float_t mone(-1.);
    Precision fSPhi = unplaced.GetStartPhiAngle();
    Precision fEPhi = unplaced.GetEPhi();
    Precision fRmax = unplaced.GetOuterRadius(); 
    Precision fRmin = unplaced.GetInnerRadius(); 
    
  // Intersection point
  Vector3D<Float_t> tmpPt;

  // General Precalcs
  Float_t rad2 = localPoint.Mag2();
  Float_t pDotV3d = localPoint.Dot(localDir);

  Float_t c = rad2 - fRmax * fRmax;
   
  //New Code
   
   Float_t sd1(kInfinity);
   Float_t d2 = (pDotV3d * pDotV3d - c);
   
   //done |= ((d2 < zero) || ((rad2 > (fRmax + halfTolerance)*(fRmax + halfTolerance)) && (pDotV3d > zero)));
   //if(IsFull(done)) return;
  
        sd1 = -1. * pDotV3d - Sqrt(d2);
        Float_t outerDist(kInfinity);
        Float_t innerDist(kInfinity);
      if(!unplaced.IsFullPhiSphere())
      {
        tmpPt.x()=localPoint.x() + sd1 * localDir.x();
        tmpPt.y()=localPoint.y() + sd1 * localDir.y();
        tmpPt.z()=localPoint.z() + sd1 * localDir.z();
        Float_t interPhi = tmpPt.Phi();
        MaskedAssign(((interPhi >= fSPhi) && (interPhi<=fEPhi)),sd1, &outerDist);
        //MaskedAssign((outerDist < zero),kInfinity, &outerDist);
      }
   
      else
      {
          outerDist = sd1;
      }
   MaskedAssign((outerDist < 0.),kInfinity, &outerDist);
   
   Precision checkRmin=unplaced.GetInnerRadius();
   
  //bool verbose=false;
  if(checkRmin)
  { 
      Float_t sd2(kInfinity);
      ////////if(verbose)std::cout<<std::endl<<"----Entered CheckRMin----"<<std::endl;
      c = rad2 - fRmin * fRmin;
      d2 = (pDotV3d * pDotV3d - c);
      sd2 = (-1. * pDotV3d + Sqrt(d2));
      if(!unplaced.IsFullPhiSphere())
      {
        tmpPt.x()=localPoint.x() + sd2 * localDir.x();
        tmpPt.y()=localPoint.y() + sd2 * localDir.y();
        tmpPt.z()=localPoint.z() + sd2 * localDir.z();
        Float_t interPhi2 = tmpPt.Phi();
        MaskedAssign(( (interPhi2 >= fSPhi) && (interPhi2<=fEPhi)),sd2, &innerDist);
        //MaskedAssign((innerDist < zero),kInfinity, &innerDist);
      }
      else
      {
          innerDist = sd2;
      }
      
      MaskedAssign((innerDist < 0.),kInfinity, &innerDist);
   
  }
   
   distance = Min(outerDist,innerDist);
   
     
   if(!unplaced.IsFullPhiSphere())
  {
      Float_t distPhi1(kInfinity);
      Float_t distPhi2(kInfinity);
      unplaced.GetWedge().DistanceToIn<Backend>(localPoint,localDir,distPhi1,distPhi2);
      Precision sinSPhi = unplaced.GetSinSPhi();
      Precision cosSPhi = unplaced.GetCosSPhi();
      Precision sinEPhi = unplaced.GetSinEPhi();
      Precision cosEPhi = unplaced.GetCosEPhi();
      
     
      Float_t distPhiMin = Min(distPhi1, distPhi2);
      Float_t distPhiMax = Max(distPhi1, distPhi2);
      Float_t minFace(1.);
      MaskedAssign((distPhi2 < distPhi1),2.,&minFace);
      
      
      tmpPt.x() = localPoint.x() + distPhiMin * localDir.x();
      tmpPt.y() = localPoint.y() + distPhiMin * localDir.y();
      tmpPt.z() = localPoint.z() + distPhiMin * localDir.z();
      
      Float_t rT = tmpPt.Mag();
      Bool_t radCond = ((rT <= fRmax) && (rT >= fRmin));
      
      CondAssign( ( radCond && (minFace == 1.)) , (tmpPt.x()*cosSPhi + tmpPt.y()*sinSPhi) ,
                (tmpPt.x()*cosEPhi + tmpPt.y()*sinEPhi),&tmpPt.x()); //&minTPoint.x());
        
      CondAssign( ( radCond && (minFace == 1.)) , (-1. *tmpPt.x()*sinSPhi + tmpPt.y()*cosSPhi) ,
                (-1. *tmpPt.x()*sinEPhi + tmpPt.y()*cosEPhi),&tmpPt.y()); //&minTPoint.y());
        
       
      Bool_t checkCond = (radCond && (tmpPt.x() > 0.));//(minTPoint.x() > zero));  
        MaskedAssign(( checkCond),Min(distance,distPhiMin), &distance);
	
                
      //Going to phiMax face
      Float_t maxFace(1.);
      MaskedAssign((distPhi2 > distPhi1),2.,&maxFace);
       
      tmpPt.x() = localPoint.x() + distPhiMax * localDir.x();
      tmpPt.y() = localPoint.y() + distPhiMax * localDir.y();
      tmpPt.z() = localPoint.z() + distPhiMax * localDir.z();
        
      rT = tmpPt.Mag();
      radCond = ((rT <= fRmax) && (rT >= fRmin));
     
      CondAssign( (radCond && (maxFace == 1.)  ) , (tmpPt.x()*cosSPhi + tmpPt.y()*sinSPhi) ,
                (tmpPt.x()*cosEPhi + tmpPt.y()*sinEPhi),&tmpPt.x());//&maxTPoint.x());
        
      CondAssign( (radCond && (maxFace == 1.)  )  , (-1. *tmpPt.x()*sinSPhi + tmpPt.y()*cosSPhi) ,
                (-1. *tmpPt.x()*sinEPhi + tmpPt.y()*cosEPhi),&tmpPt.y());//&maxTPoint.y());
        
      Bool_t checkCondMax = (radCond && (tmpPt.x() > 0.));//(maxTPoint.x() > zero));
        MaskedAssign( (!checkCond && checkCondMax) , Min(distance,distPhiMax),&distance );
      
        MaskedAssign((distance  < 0.),kInfinity, &distance);
	
   }
  
   //--------------------------------
   
    
  Bool_t thetaCond(false);
  Float_t INF(kInfinity);  
  Bool_t set(false);
  if(!unplaced.IsFullThetaSphere())
  {
      /*
    
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
   */
 // } //NOW
  
  //distance = snxt;
   
//}  //NOW

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
    typedef typename Backend::inside_v    Inside_t;

    Vector3D<Float_t> localPoint = point;
    Vector3D<Float_t> localDir = direction;
    
    distance = kInfinity;
    Bool_t done(false);
    Bool_t tr(true),fal(false);
    
    
    Precision fSPhi = unplaced.GetStartPhiAngle();
    Precision fEPhi = unplaced.GetEPhi();
    Precision fSTheta = unplaced.GetStartThetaAngle();
    Precision fETheta = unplaced.GetETheta();
    
    Float_t fRmax(unplaced.GetOuterRadius()); 
    Float_t fRmin(unplaced.GetInnerRadius()); 
    
    Precision fRminTolerance = unplaced.GetFRminTolerance()*10.;
    //Precision halfRminTolerance = 0.5 * fRminTolerance;
    Float_t halfRminTolerance = 0.5 * unplaced.GetFRminTolerance() * 10.;
    Float_t halfRmaxTolerance = 0.5 * unplaced.GetMKTolerance() * 10.;
  
    Vector3D<Float_t> tmpPt;

  
  Float_t  c(0.), d2(0.);

  // General Precalcs
  Float_t rad2 = localPoint.Mag2();
  Float_t rho2 = localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y();
  Float_t pDotV3d = localPoint.Dot(localDir);

   Bool_t cond(false);
  
   c = rad2 - fRmax * fRmax;
   
   //New Code
   
   Float_t sd1(kInfinity);
   Float_t sd2(kInfinity);
   
   bool fullPhiSphere = unplaced.IsFullPhiSphere();
   bool fullThetaSphere = unplaced.IsFullThetaSphere();
   
   MaskedAssign((tr),(pDotV3d * pDotV3d - c),&d2);
   done |= (d2 < 0. || ((localPoint.Mag() > fRmax) && (pDotV3d > 0)));
   if(IsFull(done)) return; //Returning in case of no intersection with outer shell
  
   MaskedAssign(( (Sqrt(rad2) >= (fRmax - unplaced.GetMKTolerance()) ) && (Sqrt(rad2) <= (fRmax + unplaced.GetMKTolerance())) && (pDotV3d < 0.) ),0.,&sd1);
   MaskedAssign( ( (Sqrt(rad2) > (fRmax + unplaced.GetMKTolerance()) ) && (tr) && (d2 >= 0.) && pDotV3d < 0.  ) ,(-1.*pDotV3d - Sqrt(d2)),&sd1);
  
   Float_t outerDist(kInfinity);
   Float_t innerDist(kInfinity);
   Float_t interPhi (kInfinity);
   Float_t interTheta (kInfinity);
   
   Float_t tolORMin2(0.), tolIRMin2(0.), tolORMax2(0.), tolIRMax2(0.);
   tolORMin2 = (fRmin - halfRminTolerance) * (fRmin - halfRminTolerance);
   tolIRMin2 = (fRmin + halfRminTolerance) * (fRmin + halfRminTolerance);
   
   tolORMax2 = (fRmax + halfRmaxTolerance) * (fRmax + halfRmaxTolerance);
   tolIRMax2 = (fRmax - halfRmaxTolerance) * (fRmax - halfRmaxTolerance);
   
   if(unplaced.IsFullSphere())
   {
       outerDist = sd1;
   }
   else
   { 
     //if
     //CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd1,outerDist,done);
     //outerDist =  CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd1,done);
     //MaskedAssign( c > kSTolerance * 10. * fRmax, CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd1,done) ,&outerDist);
       CondAssign( c > kSTolerance * 10. * fRmax, CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd1,done) , 
                          CheckSpecialTolerantCase<Backend>(unplaced,localPoint,localDir,false, done) , &outerDist);
     //CheckSpecialTolerantCase<Backend>();
   }
   
  if(unplaced.GetInnerRadius())
  { 
      c = rad2 - fRmin * fRmin;
      MaskedAssign(tr,(pDotV3d * pDotV3d - c),&d2);
      MaskedAssign( ( !done &&  (tr) && (d2 >= 0.) ) ,(-1.*pDotV3d + Sqrt(d2)),&sd2);
      if(unplaced.IsFullSphere())
      {
        MaskedAssign(!done , sd2, &innerDist);
      }
      else
      {
        //CheckSpecialTolerantCase<Backend>();  
        //CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd2,innerDist,done);
        //innerDist = CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd2,done);
         // MaskedAssign( c > kSTolerance * fRmax, CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd2,done) ,&innerDist);
         // Float_t tolIRMin2 = (fRmin + halfRminTolerance) * (fRmin + halfRminTolerance);
          CondAssign(((c > -1. * halfRminTolerance) && (rad2 < tolIRMin2) && ((d2 < fRmin * fRminTolerance) || (pDotV3d >= 0)) ),
                       CheckSpecialTolerantCase<Backend>(unplaced,localPoint,localDir,true, done), CheckPhiTheta<Backend>(unplaced,localPoint,localDir,sd2,done) ,
                  &innerDist);
      }
  }
   
   MaskedAssign((outerDist < 0.),kInfinity,&outerDist);
   MaskedAssign((innerDist < 0.),kInfinity,&innerDist);
   distance=Min(outerDist,innerDist);
   
   if(!fullPhiSphere)
  {
      Float_t distPhi1(kInfinity);
      Float_t distPhi2(kInfinity);
      //******************************************************
      typename Backend::inside_v onSurf;
      InsideKernel<Backend>(unplaced,localPoint,onSurf);
      Float_t one(1.);
      Float_t zero(0.);
      Float_t onSurfF(onSurf);
      Bool_t onSurfB(false);
      onSurfB = (onSurfF == one);
      //MaskedAssign(onSurfB && pDotV3d < 0.,0.,&distance);
      //if(IsFull(onSurfB))return;
      //MaskedAssign(onSurfB,0.,&distPhiMin);
      
      //******************************************************
      
      unplaced.GetWedge().DistanceToIn<Backend>(localPoint,localDir,distPhi1,distPhi2);
      Precision sinSPhi = unplaced.GetSinSPhi();
      Precision cosSPhi = unplaced.GetCosSPhi();
      Precision sinEPhi = unplaced.GetSinEPhi();
      Precision cosEPhi = unplaced.GetCosEPhi();
      
      Float_t distPhiMin = Min(distPhi1, distPhi2);
      MaskedAssign((distPhiMin<0.),kInfinity,&distPhiMin);
     
      
      Float_t distPhiMax = Max(distPhi1, distPhi2);
      MaskedAssign((distPhiMax<0.),kInfinity,&distPhiMax);
      
      Float_t minFace(1.);
      MaskedAssign((distPhi2 < distPhi1),2.,&minFace);
      
      tmpPt.x() = localPoint.x() + distPhiMin * localDir.x();
      tmpPt.y() = localPoint.y() + distPhiMin * localDir.y();
      tmpPt.z() = localPoint.z() + distPhiMin * localDir.z();
      Float_t tmpPtTheta = ATan2(Sqrt(tmpPt.x()*tmpPt.x() + tmpPt.y()*tmpPt.y()), tmpPt.z()); 
      
      Float_t rT = tmpPt.Mag();
      Bool_t radCond = ((rT <= fRmax) && (rT >= fRmin));
        
        CondAssign( ( radCond && (minFace == 1.)) , (tmpPt.x()*cosSPhi + tmpPt.y()*sinSPhi) ,
                (tmpPt.x()*cosEPhi + tmpPt.y()*sinEPhi),&tmpPt.x()); //&minTPoint.x());
        
        CondAssign( ( radCond && (minFace == 1.)) , (-1.*tmpPt.x()*sinSPhi + tmpPt.y()*cosSPhi) ,
                (-1.*tmpPt.x()*sinEPhi + tmpPt.y()*cosEPhi),&tmpPt.y()); //&minTPoint.y());
        
       Bool_t checkCond = (radCond && (tmpPt.x() > 0.));//(minTPoint.x() > zero));    
       MaskedAssign(!done &&  checkCond && (tmpPtTheta > fSTheta && tmpPtTheta < fETheta), Min(distPhiMin,distance),&distance );
       
       //Going to phiMax face
        
        Float_t maxFace(1.);
        MaskedAssign((distPhi2 > distPhi1),2.,&maxFace);
       
        MaskedAssign(!checkCond, (localPoint.x() + distPhiMax * localDir.x()),&tmpPt.x());
        MaskedAssign(!checkCond, (localPoint.y() + distPhiMax * localDir.y()),&tmpPt.y());
        MaskedAssign(!checkCond, (localPoint.z() + distPhiMax * localDir.z()),&tmpPt.z());
        
        tmpPtTheta = ATan2(Sqrt(tmpPt.x()*tmpPt.x() + tmpPt.y()*tmpPt.y()), tmpPt.z()); 
        
        rT = tmpPt.Mag();
                
        radCond = ((rT <= fRmax) && (rT >= fRmin));
        
        CondAssign( (!checkCond && radCond && (maxFace == 1.)  ) , (tmpPt.x()*cosSPhi + tmpPt.y()*sinSPhi) ,
                (tmpPt.x()*cosEPhi + tmpPt.y()*sinEPhi),&tmpPt.x());//&maxTPoint.x());
        
        CondAssign( (!checkCond && radCond && (maxFace == 1.)  )  , (-1.*tmpPt.x()*sinSPhi + tmpPt.y()*cosSPhi) ,
                (-1.*tmpPt.x()*sinEPhi + tmpPt.y()*cosEPhi),&tmpPt.y());//&maxTPoint.y());
        
        Bool_t checkCondMax = (radCond && (tmpPt.x() > 0.));//(maxTPoint.x() > zero));
        MaskedAssign(!done &&  !checkCond && checkCondMax && (tmpPtTheta > fSTheta && tmpPtTheta < fETheta), Min(distPhiMax,distance),&distance );
       
       
     
   }
 
   
   
   Float_t distThetaMin(kInfinity);
   
   if(!fullThetaSphere)
   {
      Bool_t intsect1(false);  
      Bool_t intsect2(false);  
      Float_t distTheta1(kInfinity);
      Float_t distTheta2(kInfinity);
      Vector3D<Float_t> coneIntSecPt;
      unplaced.GetThetaCone().DistanceToIn<Backend>(localPoint,localDir,distTheta1,distTheta2, intsect1,intsect2);//,cone1IntSecPt, cone2IntSecPt);
      
      MaskedAssign( (intsect1),(localPoint.x() + distTheta1 * localDir.x()),&coneIntSecPt.x());
      MaskedAssign( (intsect1),(localPoint.y() + distTheta1 * localDir.y()),&coneIntSecPt.y());
      MaskedAssign( (intsect1),(localPoint.z() + distTheta1 * localDir.z()),&coneIntSecPt.z());
      
      Float_t distCone1 = coneIntSecPt.Mag();
      Float_t phiCone1 = coneIntSecPt.Phi();
      
      
      MaskedAssign( (intsect2),(localPoint.x() + distTheta2 * localDir.x()),&coneIntSecPt.x());
      MaskedAssign( (intsect2),(localPoint.y() + distTheta2 * localDir.y()),&coneIntSecPt.y());
      MaskedAssign( (intsect2),(localPoint.z() + distTheta2 * localDir.z()),&coneIntSecPt.z());
      
      
      Float_t distCone2 = coneIntSecPt.Mag();
      Float_t phiCone2 = coneIntSecPt.Phi();
      
      Bool_t isValidCone1 = (phiCone1 > fSPhi && phiCone1  < fEPhi) && (distCone1 > fRmin && distCone1 < fRmax);
      Bool_t isValidCone2 = (phiCone2 > fSPhi && phiCone2  < fEPhi) && (distCone2 > fRmin && distCone2 < fRmax);
     
      if(!fullPhiSphere)
          {
              
            MaskedAssign( (!done && (((intsect2 && !intsect1)  && isValidCone2) || ((intsect2 && intsect1) && isValidCone2 && !isValidCone1)) ),distTheta2,&distThetaMin);
      
            MaskedAssign( (!done && (((!intsect2 && intsect1) && isValidCone1) || ((intsect2 && intsect1) && isValidCone1 && !isValidCone2)) ),distTheta1,&distThetaMin);
            
            MaskedAssign( (!done && (intsect2 && intsect1)  && isValidCone1 && isValidCone2),
                    Min(distTheta1,distTheta2),&distThetaMin);
         
          }
          else
          {
              MaskedAssign( (!done && (((intsect2 && !intsect1)  && (distCone2 > fRmin && distCone2 < fRmax)) || 
                      ((intsect2 && intsect1) &&  (distCone2 > fRmin && distCone2 < fRmax) && !(distCone1 > fRmin && distCone1 < fRmax))) ),distTheta2,&distThetaMin);
              
              MaskedAssign( (!done && (((!intsect2 && intsect1) && (distCone1 > fRmin && distCone1 < fRmax)) ||
                      ((intsect2 && intsect1) && (distCone1 > fRmin && distCone1 < fRmax) && !(distCone2 > fRmin && distCone2 < fRmax))) ),distTheta1,&distThetaMin);
              
              MaskedAssign( (!done && (intsect2 && intsect1)  && (distCone1 > fRmin && distCone1 < fRmax) && (distCone2 > fRmin && distCone2 < fRmax)),
                    Min(distTheta1,distTheta2),&distThetaMin);
      
              
          }
      
      }
   distance = Min(distThetaMin,distance);
   
   //std::cout<<"Distance : "<<distance << "  :: kSTolerance*100 : "<< kSTolerance*100. << "   ::  ( distance < kSTolerance*100.) : "<< ( distance < kSTolerance*100.)<<std::endl;
   
   MaskedAssign(( distance < kSTolerance/*2.25 * kSTolerance * 10.*/) , 0. , &distance);
   
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v SphereImplementation<transCodeT, rotCodeT>::CheckPhiTheta(UnplacedSphere const &unplaced,
                                                Vector3D<typename Backend::precision_v> const localPoint, Vector3D<typename Backend::precision_v> const localDir,
                                                typename Backend::precision_v sd, /*typename Backend::precision_v &dist,*/ typename Backend::bool_v done ){

   bool fullPhiSphere = unplaced.IsFullPhiSphere();
   bool fullThetaSphere = unplaced.IsFullThetaSphere();
   typedef typename Backend::precision_v Float_t;
   typedef typename Backend::bool_v      Bool_t;
   
   Float_t dist(kInfinity);
   
   //Float_t sd(kInfinity);
   Precision fSPhi = unplaced.GetStartPhiAngle();
   Precision fEPhi = unplaced.GetEPhi();
   Precision fSTheta = unplaced.GetStartThetaAngle();
   Precision fETheta = unplaced.GetETheta();
   
   Float_t kAngTolerance(kSTolerance);
   Float_t halfAngTolerance = (0.5 * kAngTolerance*10.);
   
   Vector3D<Float_t> tmpPt;
   
   tmpPt.x()=localPoint.x() + sd * localDir.x();
   tmpPt.y()=localPoint.y() + sd * localDir.y();
   
   if(!fullPhiSphere && fullThetaSphere)
   {  
      //tmpPt.x()=localPoint.x() + sd * localDir.x();
      //tmpPt.y()=localPoint.y() + sd * localDir.y();
      Float_t interPhi = tmpPt.Phi();
      //done |= (interPhi) && ()
      MaskedAssign((!done &&  (interPhi >= fSPhi-halfAngTolerance) && (interPhi<=fEPhi+halfAngTolerance)),sd,&dist);
   }
  else
  {
   if(fullPhiSphere && !fullThetaSphere)
   {
       tmpPt.z()=localPoint.z() + sd * localDir.z(); 
       Float_t interTheta = ATan2(Sqrt(tmpPt.x()*tmpPt.x() + tmpPt.y()*tmpPt.y()), tmpPt.z()); 
       MaskedAssign((!done &&  (interTheta >= fSTheta) && (interTheta <= fETheta)),sd,&dist);
   }
   else{
       
   if(!fullPhiSphere && !fullThetaSphere)
   {
   tmpPt.z()=localPoint.z() + sd * localDir.z();     
   Float_t interPhi = tmpPt.Phi();
   Float_t interTheta = ATan2(Sqrt(tmpPt.x()*tmpPt.x() + tmpPt.y()*tmpPt.y()), tmpPt.z());
   MaskedAssign((!done &&   ((interPhi >= fSPhi) && (interPhi<=fEPhi)) &&
                   ((interTheta >= fSTheta) && (interTheta <= fETheta)) ),sd,&dist);
   }
   }
  }
   return dist;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v SphereImplementation<transCodeT, rotCodeT>::CheckSpecialTolerantCase(UnplacedSphere const &unplaced,
                                                Vector3D<typename Backend::precision_v> const localPoint, Vector3D<typename Backend::precision_v> const localDir,
                                                bool ifRmin, typename Backend::bool_v done ){

   //bool fullPhiSphere = unplaced.IsFullPhiSphere();
   //bool fullThetaSphere = unplaced.IsFullThetaSphere();
   
   typedef typename Backend::precision_v Float_t;
   typedef typename Backend::bool_v      Bool_t;
   
   Float_t snxt(kInfinity);
   
   //General Precalc
   Float_t kAngTolerance(kSTolerance * 10.);
   Float_t halfAngTolerance = 0.5 * kAngTolerance;
   Bool_t fullThetaSphere(unplaced.IsFullThetaSphere());
   Bool_t fullPhiSphere(unplaced.IsFullPhiSphere());
   Precision fRmin = unplaced.GetInnerRadius();
   Precision fRmax = unplaced.GetOuterRadius();
   Float_t halfRminTolerance = 0.5 * unplaced.GetFRminTolerance() * 10.;
   Float_t halfRmaxTolerance = 0.5 * unplaced.GetMKTolerance() * 10.;
   Float_t tolSTheta(0.), tolETheta(0.);
   Bool_t tr(true), fal(false);
   
   
   MaskedAssign(!fullThetaSphere,(unplaced.GetStartThetaAngle() - halfAngTolerance), &tolSTheta );
   MaskedAssign(!fullThetaSphere,(unplaced.GetETheta() + halfAngTolerance), &tolETheta );
  /*
   if (!fullThetaSphere)
  {
    tolSTheta = unplaced.GetStartThetaAngle() - halfAngTolerance;
    tolETheta = unplaced.GetETheta() + halfAngTolerance;
  }
   */ 

   Float_t tolORMin2(0.), tolIRMin2(0.), tolORMax2(0.), tolIRMax2(0.);
   tolORMin2 = (fRmin - halfRminTolerance) * (fRmin - halfRminTolerance);
   tolIRMin2 = (fRmin + halfRminTolerance) * (fRmin + halfRminTolerance);
   
   tolORMax2 = (fRmax + halfRmaxTolerance) * (fRmax + halfRmaxTolerance);
   tolIRMax2 = (fRmax - halfRmaxTolerance) * (fRmax - halfRmaxTolerance);
   
   Float_t sinCPhi(unplaced.GetSinCPhi()); 
   Float_t cosCPhi(unplaced.GetCosCPhi()); 
   Float_t sinSPhi(unplaced.GetSinSPhi()); 
   Float_t cosSPhi(unplaced.GetCosSPhi()); 
   Float_t sinEPhi(unplaced.GetSinEPhi()); 
   Float_t cosEPhi(unplaced.GetCosEPhi()); 
   Float_t cosHDPhiIT(unplaced.GetCosHDPhiIT());
   Float_t cosHDPhiOT(unplaced.GetCosHDPhiOT());
   
   Float_t pDotV3d = localPoint.Dot(localDir);
   Float_t rho2 = localPoint.x() * localPoint.x() + localPoint.y() * localPoint.y();
   Float_t rad2 = localPoint.Mag2();
   Float_t c = rad2 - fRmax * fRmax;
   Float_t d2 = pDotV3d * pDotV3d - c;
   //Bool_t cond1 = ((rad2 > tolIRMax2) && ((d2 >= kTolerance * fRmax) && (pDotV3d < 0)));
   Bool_t cond1(true);
   CondAssign(ifRmin ,tr, ((rad2 > tolIRMax2) && ((d2 >= kTolerance * fRmax) && (pDotV3d < 0))) ,&cond1);
   
   Float_t cosPsi = (localPoint.x() * cosCPhi + localPoint.y() * sinCPhi) / Sqrt(rho2);
   Float_t pTheta = ATan2(Sqrt(localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y()), localPoint.z());
   
   Bool_t phiCond = cond1 && !fullPhiSphere && (cosPsi >= cosHDPhiIT) && !fullThetaSphere && ((pTheta >= tolSTheta + kAngTolerance)&& (pTheta <= tolETheta - kAngTolerance));
   MaskedAssign(!done && phiCond, 0. , &snxt );
   done |= phiCond;
   
   
   phiCond = cond1 && !fullPhiSphere && (cosPsi >= cosHDPhiIT) && fullThetaSphere;
   MaskedAssign(!done && phiCond, 0. , &snxt );
   done|=phiCond;
   
   
   Bool_t thetaCond = cond1 && fullPhiSphere && !fullThetaSphere && ((pTheta >= tolSTheta + kAngTolerance) && (pTheta <= tolETheta - kAngTolerance));
   MaskedAssign(!done && thetaCond, 0. ,&snxt);
   done |= thetaCond;
   
   Bool_t fullSphereCond = cond1 && fullPhiSphere && fullThetaSphere;
   MaskedAssign(!done && fullSphereCond, 0. , &snxt);
   done |= fullSphereCond;
   
   return snxt;
   
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

   
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Float_t> localPoint = point;
   
    Vector3D<Float_t> localDir=direction;
    
    distance = kInfinity;
    Float_t  pDotV2d, pDotV3d;
    
    
    Bool_t done(false);
    Bool_t tr(true),fal(false);
    
    Float_t snxt(kInfinity);
    
    Float_t fRmax(unplaced.GetOuterRadius()); 
    Float_t fRmin(unplaced.GetInnerRadius()); 
    Precision fSTheta = unplaced.GetStartThetaAngle();
    Precision fETheta = unplaced.GetETheta();
    
    // Intersection point
    Vector3D<Float_t> intSecPt;
    Float_t  c(0.), d2(0.);

   pDotV2d = localPoint.x() * localDir.x() + localPoint.y() * localDir.y();
   pDotV3d = pDotV2d + localPoint.z() * localDir.z(); //localPoint.Dot(localDir);

   Float_t rad2 = localPoint.Mag2();
   c = rad2 - fRmax * fRmax;
   
   //New Code
   
   Float_t sd1(kInfinity);
   Float_t sd2(kInfinity);
   Float_t Rmax_plus = fRmax + unplaced.GetMKTolerance();
   Float_t Rmin_minus = fRmin - unplaced.GetFRminTolerance();
   
   Bool_t cond1 = (Sqrt(rad2) <= (fRmax + 0.5*unplaced.GetMKTolerance())) && (Sqrt(rad2) >= (fRmin - 0.5*unplaced.GetFRminTolerance()));
   Bool_t cond = (Sqrt(rad2) <= (fRmax + unplaced.GetMKTolerance())) && (Sqrt(rad2) >= (fRmax - unplaced.GetMKTolerance())) && pDotV3d >=0 && cond1;
   done |= cond;
   MaskedAssign(cond ,0.,&sd1);
   
   MaskedAssign((tr && cond1),(pDotV3d * pDotV3d - c),&d2);
   MaskedAssign( (!done && cond1 && (tr) && (d2 >= 0.) ) ,(-1.*pDotV3d + Sqrt(d2)),&sd1);
   
   MaskedAssign((sd1 < 0.),kInfinity, &sd1);
   
   if(unplaced.GetInnerRadius())
  { 
       cond = (Sqrt(rad2) <= (fRmin + unplaced.GetFRminTolerance())) && (Sqrt(rad2) >= (fRmin - unplaced.GetFRminTolerance())) && pDotV3d < 0 && cond1;
       done |= cond;
       MaskedAssign(cond ,0.,&sd2);
      c = rad2 - fRmin * fRmin;
      
      MaskedAssign(tr,(pDotV3d * pDotV3d - c),&d2);
     
      MaskedAssign( ( !done && (tr && cond1) && (d2 >= 0.) && (pDotV3d < 0.)) ,(-1.*pDotV3d - Sqrt(d2)),&sd2);
      MaskedAssign((sd2 < 0.),kInfinity, &sd2);
      
  }
   
    snxt=Min(sd1,sd2);
    Float_t distThetaMin(kInfinity);
    Float_t distPhiMin(kInfinity);
            
    if(!unplaced.IsFullThetaSphere())
    {          
      Bool_t intsect1(false);  
      Bool_t intsect2(false);  
      Float_t distTheta1(kInfinity);
      Float_t distTheta2(kInfinity);
      unplaced.GetThetaCone().DistanceToOut<Backend>(localPoint,localDir,distTheta1,distTheta2, intsect1,intsect2);
      distThetaMin = Min(distTheta1, distTheta2);
    }
    
    
  if (!unplaced.IsFullPhiSphere())
  {
      //Using Wedge Class
      Float_t distPhi1;
      Float_t distPhi2;
      unplaced.GetWedge().DistanceToOut<Backend>(localPoint,localDir,distPhi1,distPhi2);
      distPhiMin = Min(distPhi1, distPhi2);
      
  }
    distance = Min(distThetaMin,snxt);
    distance = Min(distPhiMin,distance);
    
    MaskedAssign(( distance < kSTolerance) , 0. , &distance);
    
}


} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
