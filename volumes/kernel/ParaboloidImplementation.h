//===-- kernel/ParaboloidImplementation.h - Instruction class definition -------*- C++ -*-===//
//===----------------------------------------------------------------------===//
///
/// \file
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file implements the Paraboloid shape
///
///
/// _____________________________________________________________________________
/// A paraboloid is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
/// - the surface of revolution of a parabola described by:
/// z = a*(x*x + y*y) + b
/// The parameters a and b are automatically computed from:
/// - rlo is the radius of the circle of intersection between the
/// parabolic surface and the plane z = -dz
/// - rhi is the radius of the circle of intersection between the
/// parabolic surface and the plane z = +dz
/// -dz = a*rlo^2 + b
/// dz = a*rhi^2 + b      where: rhi>rlo, both >= 0
///
/// note:
/// dd = 1./(rhi^2 - rlo^2);
/// a = 2.*dz*dd;
/// b = - dz * (rlo^2 + rhi^2)*dd;
///
/// in respect with the G4 implementation we have:
/// k1=1/a
/// k2=-b/a
///
/// a=1/k1
/// b=-k2/k1
///
//===----------------------------------------------------------------------===//

#ifndef VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedParaboloid.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(ParaboloidImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {
    
    namespace ParaboloidUtilities
    {
        template <class Backend>
        VECGEOM_INLINE
        VECGEOM_CUDA_HEADER_BOTH
        void DistToParaboloidSurface(
                                 UnplacedParaboloid const &unplaced,
                                 Vector3D<typename Backend::precision_v> const &point,
                                 Vector3D<typename Backend::precision_v> const &direction,
                                 typename Backend::precision_v &distance/*,
                                 typename Backend::bool_v in*/) {
                                 }
    }

class PlacedParaboloid;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParaboloidImplementation {

    static const int transC = transCodeT;
    static const int rotC   = rotCodeT;

    using PlacedShape_t = PlacedParaboloid;
    using UnplacedShape_t = UnplacedParaboloid;
   
    VECGEOM_CUDA_HEADER_BOTH
    static void PrintType() {
       printf("SpecializedParaboloid<%i, %i>", transCodeT, rotCodeT);
    }

    /// \brief Inside method that takes account of the surface for an Unplaced Paraboloid
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void UnplacedInside(UnplacedParaboloid const &unplaced,
                               Vector3D<typename Backend::precision_v> point,
                               typename Backend::int_v &inside) {
        
        typedef typename Backend::precision_v Double_t;
        typedef typename Backend::bool_v      Bool_t;
        
        Double_t rho2=point.x()*point.x()+point.y()*point.y();
        
        //Check if points are above or below the solid or outside the parabolic surface
        Double_t absZ=Abs(point.z());
        Bool_t outsideAboveOrBelowOuterTolerance=(absZ > unplaced.GetTolOz());
        Bool_t isOutside= outsideAboveOrBelowOuterTolerance;
        Bool_t done(isOutside);
        
        if (Backend::early_returns && IsFull(done))
        {
            inside = EInside::kOutside;
            return;
        }
        
        Double_t value=unplaced.GetA()*rho2+unplaced.GetB()-point.z();
        Bool_t outsideParabolicSurfaceOuterTolerance= (value>kHalfTolerance);
        done|=outsideParabolicSurfaceOuterTolerance;
        if (Backend::early_returns && IsFull(done))
        {
            inside = EInside::kOutside;
            return;
        }
        
        //Check if points are inside the inner tolerance of the solid
        Bool_t insideAboveOrBelowInnerTolerance = (absZ < unplaced.GetTolOz()),
               insideParaboloidSurfaceInnerTolerance= (value<- kHalfTolerance);
        
        Bool_t isInside=insideAboveOrBelowInnerTolerance && insideParaboloidSurfaceInnerTolerance;
        MaskedAssign(isInside, EInside::kInside, &inside);
        done|=isInside;
        if(Backend::early_returns && IsFull(done)) return;
        
        MaskedAssign(!done, EInside::kSurface, &inside);
    }

    
    /// \brief UnplacedContains (ROOT STYLE): Inside method that does NOT take account of the surface for an Unplaced Paraboloid
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void UnplacedContains(UnplacedParaboloid const &unplaced,
        Vector3D<typename Backend::precision_v> point,
        typename Backend::bool_v &inside) {
        
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v      Bool_t;
        
        // //Check if points are above or below the solid
        Bool_t isAboveOrBelowSolid=(Abs(point.z()) > unplaced.GetDz());
        Bool_t done(isAboveOrBelowSolid);

        if(Backend::early_returns && IsFull(done)) return;
        
        // //Check if points are outside the parabolic surface
        Float_t aa=unplaced.GetA()*(point.z()-unplaced.GetB());
        Float_t rho2=point.x()*point.x()+point.y()*point.y();
        
        Bool_t isOutsideParabolicSurface= aa < 0 || aa < unplaced.GetA2()*rho2;
        done |= isOutsideParabolicSurface;
        
        inside = !done;
    }

    /// \brief Inside method that takes account of the surface for a Placed Paraboloid
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Inside(UnplacedParaboloid const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint,
                       typename Backend::int_v &inside) {
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedInside<Backend>(unplaced, localPoint, inside);
    }
    
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Inside(UnplacedParaboloid const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       typename Backend::int_v &inside) {
        
      Vector3D<typename Backend::precision_v> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
      UnplacedInside<Backend>(unplaced, localPoint, inside);
    }


    /// \brief Contains: Inside method that does NOT take account of the surface for a Placed Paraboloid
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Contains(UnplacedParaboloid const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint,
                       typename Backend::bool_v &inside) {
        
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedContains<Backend>(unplaced, localPoint, inside);
        
    }
    
    template <typename Backend, bool ForInside>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void GenericKernelForContainsAndInside(Vector3D<Precision> const &,
                                                  Vector3D<typename Backend::precision_v> const &,
                                                  typename Backend::bool_v &,
                                                  typename Backend::bool_v &);
    
    
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToIn(
                             UnplacedParaboloid const &unplaced,
                             Transformation3D const &transformation,
                             Vector3D<typename Backend::precision_v> const &point,
                             Vector3D<typename Backend::precision_v> const &direction,
                             typename Backend::precision_v const &stepMax,
                             typename Backend::precision_v &distance) {
        
        using namespace ParaboloidUtilities;
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v      Bool_t;
        
        
        Vector3D<Float_t> localPoint =
        transformation.Transform<transCodeT, rotCodeT>(point);
        Vector3D<Float_t> localDirection =
        transformation.TransformDirection<rotCodeT>(direction);
        
        Bool_t done(false);
        distance=kInfinity;
        
        Float_t absZ=Abs(localPoint.z());
        Float_t absDirZ=Abs(localDirection.z());
        Float_t rho2 = localPoint.x()*localPoint.x()+localPoint.y()*localPoint.y();
        Float_t point_dot_direction_x = localPoint.x()*localDirection.x();
        Float_t point_dot_direction_y = localPoint.y()*localDirection.y();
        
        Bool_t checkZ=localPoint.z()*localDirection.z() > 0;
        
        //check if the point is distancing in Z
        Bool_t isDistancingInZ= (absZ>unplaced.GetDz() && checkZ);
        done|=isDistancingInZ;
        if (Backend::early_returns && IsFull(done)) return;
        
        //check if the point is distancing in XY
        Bool_t isDistancingInXY=( (rho2>unplaced.GetRhi2()) && (point_dot_direction_x>0 && point_dot_direction_y>0) );
        done|=isDistancingInXY;
        if (Backend::early_returns && IsFull(done)) return;
    
        //check if the point is distancing in X
        Bool_t isDistancingInX=( (Abs(localPoint.x())>unplaced.GetRhi()) && (point_dot_direction_x>0) );
        done|=isDistancingInX;
        if (Backend::early_returns && IsFull(done)) return;

        //check if the point is distancing in Y
        Bool_t isDistancingInY=( (Abs(localPoint.y())>unplaced.GetRhi()) && (point_dot_direction_y>0) );
        done|=isDistancingInY;
        if (Backend::early_returns && IsFull(done)) return;

        //is hitting from dz or -dz planes
        Float_t distZ = (absZ-unplaced.GetDz())/absDirZ;
        Float_t xHit = localPoint.x()+distZ*localDirection.x();
        Float_t yHit = localPoint.y()+distZ*localDirection.y();
        Float_t rhoHit2=xHit*xHit+yHit*yHit;
        
        Float_t ray2=unplaced.GetRhi2();
        MaskedAssign(localPoint.z()<-unplaced.GetDz() && localDirection.z()>0, unplaced.GetRlo2(), &ray2); //verificare

        Bool_t isCrossingAtDz= (absZ>unplaced.GetDz()) && (!checkZ) && (rhoHit2 <=ray2);
        MaskedAssign(isCrossingAtDz, distZ, &distance);
        done|=isCrossingAtDz;
        if (Backend::early_returns && IsFull(done)) return;
        
        //is hitting from the paraboloid surface
        //DistToParaboloidSurface<Backend>(unplaced,localPoint,localDirection, distParab);
        Float_t distParab=kInfinity;
        Float_t dirRho2 = localDirection.x()*localDirection.x()+localDirection.y()*localDirection.y();
        Float_t a = unplaced.GetA() * dirRho2;
        Float_t b = 2.*unplaced.GetA()*(point_dot_direction_x+point_dot_direction_y)-localDirection.z();
        Float_t c = unplaced.GetA()*rho2 + unplaced.GetB() - localPoint.z();
        
    
        /*avoiding division per 0
        Bool_t aVerySmall=(Abs(a)<kTiny),
               bVerySmall=(Abs(b)<kTiny);
        
        done|= aVerySmall && bVerySmall;
        if (IsFull(done)) return; //big
        
        
        Double_t COverB=-c/b;
        Bool_t COverBNeg=(COverB<0);
        done|=COverBNeg && aVerySmall ; //se neg ritorno big
        if (IsFull(done)) return;
        
        Bool_t check1=aVerySmall && !bVerySmall && !COverBNeg;
        MaskedAssign(!done && check1 , COverB, &distParab ); //to store
        */
        
        Float_t ainv = 1./a;
        Float_t t = b*0.5;
        Float_t prod = c*a;
        Float_t delta = t*t - prod;
        
        Bool_t deltaNeg=delta<0;
        done|= deltaNeg;
        if (Backend::early_returns && IsFull(done)) return;
        
        //to avoid square root operation on negative elements
        MaskedAssign(deltaNeg, 0. , &delta);
        delta = Sqrt(delta);


        //I take only the biggest solution among all
        distParab=ainv*(-t - delta);
        
        Float_t zHit = localPoint.z()+distParab*localDirection.z();
        Bool_t isHittingParaboloidSurface = ( (distParab > 1E20) || (Abs(zHit)<=unplaced.GetDz()) ); //why: dist > 1E20?
        MaskedAssign(!done && isHittingParaboloidSurface && !deltaNeg && distParab>0 , distParab, &distance);
        
    }
    
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToOut(
                              UnplacedParaboloid const &unplaced,
                              Vector3D<typename Backend::precision_v> point,
                              Vector3D<typename Backend::precision_v> direction,
                              typename Backend::precision_v const &stepMax,
                              typename Backend::precision_v &distance) {
      
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v      Bool_t;
        
        
        
        Float_t distZ=kInfinity;
        Float_t dirZinv=1/direction.z();
        
        Bool_t dir_mask= direction.z()<0;
        MaskedAssign(dir_mask, -(unplaced.GetDz() + point.z())*dirZinv, &distZ);
        MaskedAssign(!dir_mask, (unplaced.GetDz() - point.z())*dirZinv, &distZ);
        
        Float_t distParab=kInfinity;
        Float_t rho2 = point.x()*point.x()+point.y()*point.y();
        Float_t dirRho2 = direction.x()*direction.x()+direction.y()*direction.y();
        Float_t a = unplaced.GetA() * dirRho2;
        Float_t b = 2.*unplaced.GetA()*(point.x()*direction.x()+point.y()*direction.y())-direction.z();
        Float_t c = unplaced.GetA()*rho2 + unplaced.GetB() - point.z();
        
        Float_t ainv = 1./a;
        Float_t t = b*0.5;
        Float_t prod = c*a;
        Float_t delta = t*t - prod;
    
        //to avoid square root operation on negative element
        //MaskedAssign(deltaNeg, 0. , &delta);
        //But if the point is inside the solid, delta cannot be negative
        delta = Sqrt(delta);
        
        Bool_t mask_sign=(ainv<0);
        Float_t sign=1.;
        MaskedAssign(mask_sign, -1., &sign);
        
        Float_t d1=ainv*(-t - sign*delta);
        Float_t d2=ainv*(-t + sign*delta);
    
        //MaskedAssign(!deltaNeg && d1>0 , d1, &distParab);
        //MaskedAssign(!deltaNeg && d1<0 && d2>0 , d2, &distParab);
        //MaskedAssign(deltaNeg || (d1<0 && d2<0), kInfinity, &distParab);
        
        MaskedAssign(d1>0 , d1, &distParab);
        MaskedAssign(d1<0 && d2>0 , d2, &distParab);
        MaskedAssign(d1<0 && d2<0, kInfinity, &distParab);
        
        distance=Min(distParab, distZ);

    }

    template<class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToIn(UnplacedParaboloid const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
        
        typedef typename Backend::precision_v Float_t;
        
        
        Vector3D<Float_t> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        
        safety=0.;
        Float_t safety_t;
        Float_t absZ= Abs(localPoint.z());
        Float_t safeZ= absZ-unplaced.GetDz();

#ifdef FAST1

        //FAST implementation starts here -- > v.1.
        //this version give 0 if the points is between the bounding box and the solid
       
        Float_t absX= Abs(localPoint.x());
        Float_t absY= Abs(localPoint.y());
        Float_t safeX= absX-unplaced.GetRhi();
        Float_t safeY= absY-unplaced.GetRhi();
        
        safety_t=Max(safeX, safeY);
        safety_t=Max(safety_t, safeZ);
        MaskedAssign(safety_t>0, safety_t, &safety);
        //FAST implementation v.1 ends here
        
#endif

        
#ifdef FAST2
        //FAST implementation starts here -- > v.2
        //this version give 0 if the points is between the bounding cylinder and the solid
        
        Float_t r=Sqrt(localPoint.x()*localPoint.x()+localPoint.y()*localPoint.y());
        Float_t safeRhi=r-unplaced.GetRhi();
    
        safety_t=Max(safeZ, safeRhi);
        MaskedAssign(safety_t>0, safety_t, &safety);
        //FAST implementation v.2 ends here
        
#endif
        
        
#ifdef ACCURATE1
        //ACCURATE bounding box implementation starts here -- > v.1
        //if the point is outside the bounding box-> FAST, otherwise calculate tangent
        
        typedef typename Backend::bool_v      Bool_t;
        
        Float_t absX= Abs(localPoint.x());
        Float_t absY= Abs(localPoint.y());
        Float_t safeX= absX-unplaced.GetRhi();
        Float_t safeY= absY-unplaced.GetRhi();
       
        Bool_t mask_bb= (safeX>0) || (safeY>0) || (safeZ>0);
        
        safety_t=Max(safeX, safeY);
        safety_t=Max(safeZ, safety_t);
        
        MaskedAssign(mask_bb , safety_t, &safety);
        if (Backend::early_returns && IsFull(mask_bb)) return;
        
        //then go for the tangent
        Float_t r0sq = (localPoint.z() - unplaced.GetB())*unplaced.GetAinv();
        Float_t r=Sqrt(localPoint.x()*localPoint.x()+localPoint.y()*localPoint.y());
        Float_t dr = r-Sqrt(r0sq);
        Float_t talf = -2.*unplaced.GetA()*Sqrt(r0sq);
        Float_t salf = talf/Sqrt(1.+talf*talf);
        Float_t safR = Abs(dr*salf);
        
        Float_t max_safety= Max(safR, safeZ);
        MaskedAssign(!mask_bb, max_safety, &safety);
        //ACCURATE implementation v.1 ends here
        
#endif
      
//#ifdef  DEFAULT
        //ACCURATE bounding cylinder implementation starts here -- > v.2

        //if the point is outside the bounding cylinder --> FAST, otherwise calculate tangent
        
        typedef typename Backend::bool_v      Bool_t;
        
        Float_t r=Sqrt(localPoint.x()*localPoint.x()+localPoint.y()*localPoint.y());
        Float_t safeRhi=r-unplaced.GetRhi();
        
        Bool_t mask_bc= (safeZ>0) || (safeRhi>0);
        safety_t=Max(safeZ, safeRhi);
        
        MaskedAssign(mask_bc, safety_t, &safety);
        if (Backend::early_returns && IsFull(mask_bc)) return;
        
        //then go for the tangent
        Float_t r0sq = (localPoint.z() - unplaced.GetB())*unplaced.GetAinv();
        Float_t dr = r-Sqrt(r0sq);
        Float_t talf = -2.*unplaced.GetA()*Sqrt(r0sq);
        Float_t salf = talf/Sqrt(1.+talf*talf);
        Float_t safR = Abs(dr*salf);
        
        Float_t max_safety= Max(safR, safeZ);
        MaskedAssign(!mask_bc, max_safety, &safety);
        //ACCURATE implementation v.2 ends here
//#endif


#ifdef ACCURATEROOTLIKE
        //ACCURATE "root-like" implementation starts here
        Float_t safZ = (Abs(localPoint.z()) - unplaced.GetDz());

        Float_t r0sq = (localPoint.z() - unplaced.GetB())*unplaced.GetAinv();
        
        Bool_t done(false);
        safety = safeZ;
        
        Bool_t underParaboloid = (r0sq<0);
        done|= underParaboloid;
        if (Backend::early_returns && IsFull(done)) return;


        Float_t safR=kInfinity;
        Float_t ro2=localPoint.x()*localPoint.x()+localPoint.y()*localPoint.y();
        Float_t z0= unplaced.GetA()*ro2+unplaced.GetB();
        Float_t dr=Sqrt(ro2)-Sqrt(r0sq);

        
        Bool_t drCloseToZero = (dr<1.E-8);
        done|=drCloseToZero;
        if (Backend::early_returns && IsFull(done)) return;
        
        //then go for the tangent
        Float_t talf = -2.*unplaced.GetA()*Sqrt(r0sq);
        Float_t salf = talf/Sqrt(1.+talf*talf);
        safeR = Abs(dr*salf);
        
        Float_t max_safety= Max(safeR, safeZ);
        MaskedAssign(!done, max_safety, &safety);
        //ACCURATE "root-like" implementation ends here
#endif
    }
    
    template<class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToOut(UnplacedParaboloid const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {

        typedef typename Backend::precision_v Float_t;
       
        
        
    
        Float_t absZ= Abs(point.z());
        Float_t safZ=(unplaced.GetDz() - absZ);

        //FAST implementation starts here
        safety=0.;
        Float_t safety_t;
        Float_t r=Sqrt(point.x()*point.x()+point.y()*point.y());
        Float_t safRlo=unplaced.GetRlo() - r;
        safety_t=Min(safZ, safRlo);
        MaskedAssign(safety_t>0, safety_t, &safety);
        
        //FAST implementation ends here
        

#if 0
        //ACCURATE implementation starts here
        typedef typename Backend::bool_v      Bool_t;
        Float_t r0sq = (point.z() - unplaced.GetB())*unplaced.GetAinv();
        
        Bool_t done(false);
        safety=0.;
        
        Bool_t closeToParaboloid = (r0sq<0);
        done|= closeToParaboloid;
        if (Backend::early_returns && IsFull(done)) return;
        
        Float_t safR=kInfinity;
        Float_t ro2=point.x()*point.x()+point.y()*point.y();
        Float_t z0= unplaced.GetA()*ro2+unplaced.GetB();
        Float_t dr=Sqrt(ro2)-Sqrt(r0sq); //avoid square root of a negative number
        
        Bool_t drCloseToZero= (dr>-1.E-8);
        done|=drCloseToZero;
        if (Backend::early_returns && IsFull(done)) return;
        
        Float_t dz = Abs(point.z()-z0);
        safR = -dr*dz/Sqrt(dr*dr+dz*dz);
        
        Float_t min_safety=Min(safR, safZ);
        MaskedAssign(!done, min_safety, &safety);
        //ACCURATE implementation ends here
#endif
    }

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_
