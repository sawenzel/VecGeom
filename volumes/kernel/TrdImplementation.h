//
/// @file TrdImplementation.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_


#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/kernel/shapetypes/TrdTypes.h"
#include <stdlib.h>
#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(TrdImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric, typename)

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace TrdUtilities {

/*
 * Checks whether a point (x, y) falls on the left or right half-plane
 * of a line. The line is defined by a (vx, vy) vector, extended to infinity.
 *
 * Of course this can only be used for lines that pass through (0, 0), but
 * you can supply transformed coordinates for the point to check for any line.
 *
 * This simply calculates the magnitude of the cross product of vectors (px, py) 
 * and (vx, vy), which is defined as |x| * |v| * sin theta.
 *
 * If the cross product is positive, the point is clockwise of V, or the "right"
 * half-plane. If it's negative, the point is CCW and on the "left" half-plane. 
 */

template<typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PointLineOrientation(typename Backend::precision_v const &px, typename Backend::precision_v const &py,
                          Precision const &vx, Precision const &vy,
                          typename Backend::precision_v &crossProduct) {
  crossProduct = vx * py - vy * px;
}
/*
 * Check intersection of the trajectory of a particle with a segment 
 * that's bound from -Ylimit to +Ylimit.j
 *
 * All points of the along-vector of a plane lie on
 * s * (alongX, alongY)
 * All points of the trajectory of the particle lie on
 * (x, y) + t * (vx, vy)
 * Thefore, it must hold that s * (alongX, alongY) == (x, y) + t * (vx, vy)
 * Solving by t we get t = (alongY*x - alongX*y) / (vy*alongX - vx*alongY)
 * 
 * t gives the distance, but how to make sure hitpoint is inside the
 * segment and not just the infinite line defined by the segment?
 *
 * Check that |hity| <= Ylimit
 */
  
template<typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PlaneTrajectoryIntersection(typename Backend::precision_v const &alongX,
        typename Backend::precision_v const &alongY,
        typename Backend::precision_v const &ylimit,
        typename Backend::precision_v const &posx,
        typename Backend::precision_v const &posy,
        typename Backend::precision_v const &dirx,
        typename Backend::precision_v const &diry,
        typename Backend::precision_v &dist,
        typename Backend::bool_v &ok) {
  typedef typename Backend::precision_v Float_t;

  dist = (alongY*posx - alongX*posy ) / (diry*alongX - dirx*alongY);
  
  Float_t hity = posy + dist*diry;
  ok = Abs(hity) <= ylimit && dist > 0;
}

template<typename Backend, bool forY, bool mirroredPoint>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void FaceTrajectoryIntersection(UnplacedTrd const &trd,
                                Vector3D<typename Backend::precision_v> const &pos,
                                Vector3D<typename Backend::precision_v> const &dir,
                                typename Backend::precision_v &dist,
                                typename Backend::bool_v &ok) {
    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v Bool_t;
    Float_t alongV, posV, dirV, posK, dirK, fK, halfKplus, v1;
    if(forY) {
        alongV = trd.y2minusy1();
        v1 = trd.dy1();
        posV = pos.y();
        posK = pos.x();
        dirV = dir.y();
        dirK = dir.x();
        fK = trd.fx();
        halfKplus = trd.halfx1plusx2();
        //ok = pos.y() * dir.y() <= 0;
    }
    else {
        alongV = trd.x2minusx1();
        v1 = trd.dx1();
        posV = pos.x();
        posK = pos.y();
        dirV = dir.x();
        dirK = dir.y();
        fK = trd.fy();
        halfKplus = trd.halfy1plusy2();
        //ok = pos.x() * dir.x() <= 0;
    }
    if(mirroredPoint) {
        posV *= -1;
        dirV *= -1;
    }
    Float_t alongZ = trd.dztimes2();

    // distance from trajectory to face
    dist = (alongZ*(posV-v1) - alongV*(pos.z()+trd.dz())  ) / (dir.z()*alongV - dirV*alongZ);
    ok = dist > 0;
    if(ok != Backend::kFalse) {
      // need to make sure z hit falls within bounds
      Float_t hitz = pos.z() + dist*dir.z();
      ok &= Abs(hitz) <= trd.dz();
      // need to make sure hit on varying dimension falls within bounds
      Float_t hitk = posK + dist*dirK;
      Float_t dK = halfKplus - fK * hitz; // calculate the width of the varying dimension at hitz
      ok &= Abs(hitk) <= dK;
    }
}

template<typename Backend, typename trdTypeT, bool inside>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Safety(UnplacedTrd const &trd, 
            Vector3D<typename Backend::precision_v> const &pos,
            typename Backend::precision_v &dist) {
    using namespace TrdTypes;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t safz = trd.dz() - Abs(pos.z());
    //std::cout << "safz: " << safz << std::endl;
    dist = safz;

    Float_t distx = trd.halfx1plusx2()-trd.fx()*pos.z();
    Bool_t okx = distx >= 0;
    Float_t safx = (distx-Abs(pos.x()))*trd.calfx();
    MaskedAssign(okx && safx < dist, safx, &dist);
    //std::cout << "safx: " << safx << std::endl;

    if(checkVaryingY<trdTypeT>(trd)) {
      Float_t disty = trd.halfy1plusy2()-trd.fy()*pos.z();
      Bool_t oky = disty >= 0;
      Float_t safy = (disty-Abs(pos.y()))*trd.calfy();
      MaskedAssign(oky && safy < dist, safy, &dist);
    }
    else {
      Float_t safy = trd.dy1() - Abs(pos.y());
      MaskedAssign(safy < dist, safy, &dist);
    }
    if(!inside) dist = -dist;
}

template <typename Backend, typename trdTypeT, bool surfaceT>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void UnplacedInside(
        UnplacedTrd const &trd,
        Vector3D<typename Backend::precision_v> const &point,
        typename Backend::bool_v &completelyinside,
        typename Backend::bool_v &completelyoutside) {

    using namespace TrdUtilities;
    using namespace TrdTypes;  
    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v Bool_t;

    Float_t pzPlusDz = point.z()+trd.dz();

    // inside Z?
    completelyoutside = Abs(point.z()) > MakePlusTolerant<surfaceT>(trd.dz());
    if(surfaceT) completelyinside = Abs(point.z()) < MakeMinusTolerant<surfaceT>(trd.dz());

    // inside X?
    Float_t cross;
    PointLineOrientation<Backend>(Abs(point.x()) - trd.dx1(), pzPlusDz, trd.x2minusx1(), trd.dztimes2(), cross);
    completelyoutside |= MakePlusTolerant<surfaceT>(cross) < Backend::kZero;
    if(surfaceT) completelyinside &= MakeMinusTolerant<surfaceT>(cross) > Backend::kZero;

    if(HasVaryingY<trdTypeT>::value != TrdTypes::kNo) {
        // If Trd type is unknown don't bother with a runtime check, assume
        // the general case
        PointLineOrientation<Backend>(Abs(point.y()) - trd.dy1(), pzPlusDz, trd.y2minusy1(), trd.dztimes2(), cross);
        completelyoutside |= MakePlusTolerant<surfaceT>(cross) < 0;
        if(surfaceT) completelyinside &= MakeMinusTolerant<surfaceT>(cross) > 0;
    }
    else {
        completelyoutside |= Abs(point.y()) > MakePlusTolerant<surfaceT>(trd.dy1());
        if(surfaceT) completelyinside &= Abs(point.y()) < MakeMinusTolerant<surfaceT>(trd.dy1());
    }
}



} // Trd utilities


class PlacedTrd;

template <TranslationCode transCodeT, RotationCode rotCodeT, typename trdTypeT>
struct TrdImplementation {

#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL int transC = transCodeT;
  VECGEOM_GLOBAL int rotC   = rotCodeT;
#else
  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;
#endif

  using PlacedShape_t = PlacedTrd;
  using UnplacedShape_t = UnplacedTrd;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedTrd<%i, %i, %s>", transCodeT, rotCodeT, trdTypeT::toString());
  }

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedTrd const &trd,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {

    typename Backend::bool_v unused;
    TrdUtilities::UnplacedInside<Backend, trdTypeT, false>(trd, point, unused, inside);
    inside = !inside;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedTrd const &trd,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {
    
    typename Backend::bool_v unused;
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    TrdUtilities::UnplacedInside<Backend, trdTypeT, false>(trd, localPoint, unused, inside);
    inside = !inside;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTrd const &trd,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside) {
    typedef typename Backend::bool_v Bool_t;
    Vector3D<typename Backend::precision_v> localpoint;
    localpoint = transformation.Transform<transCodeT, rotCodeT>(point);
    inside = EInside::kOutside;

    Bool_t completelyoutside, completelyinside;
    TrdUtilities::UnplacedInside<Backend, trdTypeT, true>(trd, localpoint, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    MaskedAssign(completelyinside, EInside::kInside, &inside);
    MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedTrd const &trd,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    using namespace TrdUtilities;
    using namespace TrdTypes;
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;

    Float_t hitx, hity;
    // Float_t hitz;

    Vector3D<Float_t> pos_local;
    Vector3D<Float_t> dir_local;
    distance = kInfinity;

    transformation.Transform<transCodeT, rotCodeT>(point, pos_local);
    transformation.TransformDirection<rotCodeT>(direction, dir_local);
    
    // hit Z faces?
    Bool_t okz = pos_local.z() * dir_local.z() < 0;
    if(okz != Backend::kFalse) {
      Float_t distz = (Abs(pos_local.z()) - trd.dz()) / Abs(dir_local.z());
      // exclude case in which particle is going away
      hitx = Abs(pos_local.x() + distz*dir_local.x());
      hity = Abs(pos_local.y() + distz*dir_local.y());

      // hitting top face?
      Bool_t okzt = pos_local.z() > trd.dz() && hitx <= trd.dx2() && hity <= trd.dy2();
      // hitting bottom face?
      Bool_t okzb = pos_local.z() < - trd.dz() && hitx <= trd.dx1() && hity <= trd.dy1();
     
      okz &= (okzt | okzb);
      MaskedAssign(okz, distz, &distance);
    }

    // hitting X faces?
    Float_t distx;
    Bool_t okx;

    FaceTrajectoryIntersection<Backend, false, false>(trd, pos_local, dir_local, distx, okx);
    MaskedAssign(okx && distx < distance, distx, &distance);
      
    FaceTrajectoryIntersection<Backend, false, true>(trd, pos_local, dir_local, distx, okx);
    MaskedAssign(okx && distx < distance, distx, &distance);
      
    // hitting Y faces?
    Float_t disty;
    Bool_t oky;

    if(checkVaryingY<trdTypeT>(trd)) {
      FaceTrajectoryIntersection<Backend, true, false>(trd, pos_local, dir_local, disty, oky);
      MaskedAssign(oky && disty < distance, disty, &distance);

      FaceTrajectoryIntersection<Backend, true, true>(trd, pos_local, dir_local, disty, oky);
      MaskedAssign(oky && disty < distance, disty, &distance);
    }
    else {
      disty = (Abs(pos_local.y()) - trd.dy1()) / Abs(dir_local.y());
      Float_t zhit = pos_local.z() + disty*dir_local.z();
      Float_t xhit = pos_local.x() + disty*dir_local.x();
      Float_t dx = trd.halfx1plusx2() - trd.fx()*zhit;
      oky = pos_local.y()*dir_local.y() < 0 && disty > 0 && Abs(xhit) < dx && Abs(zhit) < trd.dz();
      MaskedAssign(oky && disty < distance, disty, &distance);
    }
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTrd const &trd,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {
    
    using namespace TrdUtilities;
    using namespace TrdTypes;
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;
    
    Float_t hitx, hity;
    // Float_t hitz;
    // Bool_t done = Backend::kFalse;
    distance = kInfinity;

    // hit top Z face?
    Bool_t okzt = dir.z() > 0;
    if(okzt != Backend::kFalse) {
      Float_t distz = (trd.dz() - point.z()) / Abs(dir.z());
      hitx = Abs(point.x() + distz*dir.x());
      hity = Abs(point.y() + distz*dir.y());
      Bool_t okzt = dir.z() > 0;
      okzt &= hitx <= trd.dx2() && hity <= trd.dy2();
      MaskedAssign(okzt, distz, &distance);
      if(Backend::early_returns && okzt == Backend::kTrue) return;
    }

    // hit bottom Z face?
    Bool_t okzb = dir.z() < 0;
    if(okzb != Backend::kFalse) {
      Float_t distz = (point.z() - (-trd.dz()) ) / Abs(dir.z());
      hitx = Abs(point.x() + distz*dir.x());
      hity = Abs(point.y() + distz*dir.y());
      Bool_t okzb = dir.z() < 0;
      okzb &= hitx <= trd.dx1() && hity <= trd.dy1();
      MaskedAssign(okzb, distz, &distance);
      if(Backend::early_returns && okzb == Backend::kTrue) return;
    }

    // hitting X faces?
    Float_t distx;
    Bool_t okx;

    FaceTrajectoryIntersection<Backend, false, false>(trd, point, dir, distx, okx);
    
    MaskedAssign(okx, distx, &distance);
    if(Backend::early_returns && okx == Backend::kTrue) return;
      
    FaceTrajectoryIntersection<Backend, false, true>(trd, point, dir, distx, okx);
    MaskedAssign(okx, distx, &distance);
    if(Backend::early_returns && okx == Backend::kTrue) return;

    // hitting X faces?
    Float_t disty;
    Bool_t oky;

    if(checkVaryingY<trdTypeT>(trd)) {
      FaceTrajectoryIntersection<Backend, true, false>(trd, point, dir, disty, oky);
      MaskedAssign(oky, disty, &distance);
      if(Backend::early_returns && oky == Backend::kTrue) return;
      
      FaceTrajectoryIntersection<Backend, true, true>(trd, point, dir, disty, oky);
      MaskedAssign(oky, disty, &distance);
    }
    else {
      Float_t plane = trd.dy1();
      MaskedAssign(dir.y() < 0, -trd.dy1(), &plane);
      disty = (plane - point.y()) / dir.y();
      Float_t zhit = point.z() + disty*dir.z();
      Float_t xhit = point.x() + disty*dir.x();
      Float_t dx = trd.halfx1plusx2() - trd.fx()*zhit;
      oky = Abs(xhit) < dx && Abs(zhit) < trd.dz();
      MaskedAssign(oky, disty, &distance);
    }
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTrd const &trd,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
    using namespace TrdUtilities;
    typedef typename Backend::precision_v Float_t;
    Vector3D<Float_t> pos_local;
    transformation.Transform<transCodeT, rotCodeT>(point, pos_local);
    Safety<Backend, trdTypeT, false>(trd, pos_local, safety);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTrd const &trd,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {
    using namespace TrdUtilities;
    Safety<Backend, trdTypeT, true>(trd, point, safety); 
  }

};

} } // End global namespace


#endif // VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
