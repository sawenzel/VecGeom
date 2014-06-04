//===-- kernel/ParaboloidImplementation.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
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

#include "base/global.h"

#include "base/transformation3d.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedParaboloid.h"

namespace VECGEOM_NAMESPACE {
    
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

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParaboloidImplementation {

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
        if (done == Backend::kTrue)
        {
            inside = EInside::kOutside;
            return;
        }
        
        Double_t value=unplaced.GetA()*rho2+unplaced.GetB()-point.z();
        
        Bool_t outsideParabolicSurfaceOuterTolerance= (value>kHalfTolerance);
        done|=outsideParabolicSurfaceOuterTolerance;
        if (done == Backend::kTrue)
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
        if(done == Backend::kTrue) return;
        
        MaskedAssign(!done, EInside::kSurface, &inside);
    }

    
    /// \brief UnplacedContains (ROOT STYLE): Inside method that does NOT take account of the surface for an Unplaced Paraboloid
#if 0
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void UnplacedContains(UnplacedParaboloid const &unplaced,
        Vector3D<typename Backend::precision_v> point,
        typename Backend::bool_v &inside) {
        
        typedef typename Backend::precision_v Double_t;
        typedef typename Backend::bool_v      Bool_t;
        
        //Check if points are above or below the solid
        Bool_t isAboveOrBelowSolid=(Abs(point.z()) > unplaced.GetDz());
        //done|=isAboveOrBelowSolid;
        //if (done == Backend::kTrue) return;
        
        inside = EInside::kOutside;
        if(Backend::early_returns && isAboveOrBelowSolid) return;
        
        Bool_t done(isAboveOrBelowSolid);
        
        //Check if points are outside the parabolic surface
        Double_t aa=unplaced.GetA()*(point.z()-unplaced.GetB());
        Double_t rho2=point.x()*point.x()+point.y()*point.y();
        
        Bool_t isOutsideParabolicSurface= aa <0 || aa<unplaced.GetA2()*rho2;
        done |= isOutsideParabolicSurface;
        
        MaskedAssign(!done, EInside::kInside, &inside);
    }

#endif
    
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
    
#if 0
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
        UnplacedInside<Backend>(unplaced, localPoint, inside);
        
    }

#endif
    
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
        
        Float_t absZ=Abs(localPoint.z()),
                absDirZ=Abs(localDirection.z()),
                rho2 = localPoint.x()*localPoint.x()+localPoint.y()*localPoint.y(),
                point_dot_direction_x = localPoint.x()*localDirection.x(),
                point_dot_direction_y = localPoint.y()*localDirection.y();
        
        Bool_t checkZ=localPoint.z()*localDirection.z() > 0;
        
        //check if the point is distancing in Z
        Bool_t isDistancingInZ= (absZ>unplaced.GetDz() && checkZ);
        done|=isDistancingInZ;
        if (done == Backend::kTrue) return;
        
        //check if the point is distancing in XY
        Bool_t isDistancingInXY=( (rho2>unplaced.GetRhi2()) && (point_dot_direction_x>0 || point_dot_direction_y>0) );
        done|=isDistancingInXY;
        if (done == Backend::kTrue) return;
    
        //is hitting from dz or -dz planes
        Float_t xHit, yHit, zHit, rhoHit2, distZ, ray2;
        
        distZ = (absZ-unplaced.GetDz())/absDirZ;
        xHit = localPoint.x()+distZ*localDirection.x();
        yHit = localPoint.y()+distZ*localDirection.y();
        rhoHit2=xHit*xHit+yHit*yHit;
        
        ray2=unplaced.GetRhi2();
        MaskedAssign(localPoint.z()<-unplaced.GetDz() && localDirection.z()>0, unplaced.GetRlo2(), &ray2); //verificare

        Bool_t isCrossingAtDz= (absZ>unplaced.GetDz()) && (!checkZ) && (rhoHit2 <=ray2);
        MaskedAssign(isCrossingAtDz, distZ, &distance);
        done|=isCrossingAtDz;
        if (done == Backend::kTrue) return;
        
        //is hitting from the paraboloid surface
        //DistToParaboloidSurface<Backend>(unplaced,localPoint,localDirection, distParab);
        
        
        Float_t distParab=kInfinity,
                dirRho2 = localDirection.x()*localDirection.x()+localDirection.y()*localDirection.y(),
                a = unplaced.GetA() * dirRho2,
                b = 2.*unplaced.GetA()*(point_dot_direction_x+point_dot_direction_y)-localDirection.z(),
                c = unplaced.GetA()*rho2 + unplaced.GetB() - localPoint.z();
        
    
         /*avoiding division per 0
        Bool_t aVerySmall=(Abs(a)<kTiny),
               bVerySmall=(Abs(b)<kTiny);
        
        done|= aVerySmall && bVerySmall;
        if (done == Backend::kTrue) return; //big
        
        
        Double_t COverB=-c/b;
        Bool_t COverBNeg=(COverB<0);
        done|=COverBNeg && aVerySmall ; //se neg ritorno big
        if (done == Backend::kTrue) return;
        
        Bool_t check1=aVerySmall && !bVerySmall && !COverBNeg;
        MaskedAssign(!done && check1 , COverB, &distParab ); //da memorizzare dopo
          */
        
        Float_t ainv = 1./a,
        t = b*0.5,
        prod = c*a,
        delta = t*t - prod;
        
        Bool_t deltaNeg=delta<0;
        done|= deltaNeg;
        if (done == Backend::kTrue) return;
        
        //to avoid square root operation on negative elements
        MaskedAssign(deltaNeg, 0. , &delta);
        delta = Sqrt(delta);
        distParab=ainv*(-t - delta);
        
        zHit = localPoint.z()+distParab*localDirection.z();
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
      // NYI
    }

    template<class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToIn(UnplacedParaboloid const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
        // NYI
    }
    template<class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToOut(UnplacedParaboloid const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {
        // NYI
    }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_