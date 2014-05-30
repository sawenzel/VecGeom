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

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ParaboloidImplementation {

    /// \brief Inside method that takes account of the surface for an Unplaced Paraboloid
    template <class Backend>
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
//#if 0
    template <class Backend>
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

//#endif
    
    /// \brief Inside method that takes account of the surface for a Placed Paraboloid
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    static void Inside(UnplacedParaboloid const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint,
                       typename Backend::int_v &inside) {
        
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedInside<Backend>(unplaced, localPoint, inside);
        
    }
    
//#if 0
    /// \brief Contains: Inside method that does NOT take account of the surface for a Placed Paraboloid
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    static void Contains(UnplacedParaboloid const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint,
                       typename Backend::bool_v &inside) {
        
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedInside<Backend>(unplaced, localPoint, inside);
        
    }

//#endif
    
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToIn(
                             UnplacedParaboloid const &unplaced,
                             Transformation3D const &transformation,
                             Vector3D<typename Backend::precision_v> const &point,
                             Vector3D<typename Backend::precision_v> const &direction,
                             typename Backend::precision_v const &stepMax,
                             typename Backend::precision_v &distance) {
    
        //NYI
    }


  template <class Backend>
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
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedParaboloid const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
    // NYI
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedParaboloid const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {
    // NYI
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_