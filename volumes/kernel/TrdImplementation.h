//
/// @file TrdImplementation.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_


#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/kernel/shapetypes/TrdTypes.h"

#include <stdlib.h>

namespace VECGEOM_NAMESPACE {

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
    typedef typename Backend::bool_v Bool_t;
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
    //Bool_t flip = posV < 0;
    //posV = Abs(posV);
    //MaskedAssign(flip, -dirV, &dirV);
    Float_t alongZ = trd.dztimes2();

    // distance from trajectory to face
    //std::cout << "alongZ = " << alongZ << std::endl;
    //std::cout << "alongV = " << alongV << std::endl;
    //std::cout << "posV = " << posV << std::endl;
    //std::cout << "dirV = " << dirV << std::endl;
    dist = (alongZ*(posV-v1) - alongV*(pos.z()+trd.dz())  ) / (dir.z()*alongV - dirV*alongZ);
    //std::cout << "candidate dist = " << dist << " ( Y = " << forY << " ) " << std::endl;

    // need to make sure z hit falls within bounds
    Float_t hitz = pos.z() + dist*dir.z();
    ok = Abs(hitz) <= trd.dz() && dist > 0;

    // need to make sure hit on varying dimension falls within bounds
    Float_t hitk = posK + dist*dirK;
    Float_t dK = halfKplus - fK * hitz; // calculate the width of the varying dimension at hitz
    ok &= Abs(hitk) <= dK;
}

template<typename Backend, bool inside>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Safety(UnplacedTrd const &trd, 
            Vector3D<typename Backend::precision_v> const &pos,
            typename Backend::precision_v &dist) {
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

    Float_t disty = trd.halfy1plusy2()-trd.fy()*pos.z();
    Bool_t oky = disty >= 0;
    Float_t safy = (disty-Abs(pos.y()))*trd.calfy();
    MaskedAssign(oky && safy < dist, safy, &dist);
    //std::cout << "safy: " << safy << std::endl;
    if(!inside) dist = -dist;
}

} // Trd utilities

template <TranslationCode transCodeT, RotationCode rotCodeT, typename trdTypeT>
struct TrdImplementation {

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedTrd const &trd,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {

    using namespace TrdUtilities;
    using namespace TrdTypes;  
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t pzPlusDz = point.z()+trd.dz();

    // inside Z?
    Bool_t inz = Abs(point.z()) <= trd.dz();

    // inside X?
    Float_t cross;
    PointLineOrientation<Backend>(Abs(point.x()) - trd.dx1(), pzPlusDz, trd.x2minusx1(), trd.dztimes2(), cross);
    Bool_t inx = cross >= 0;

    Bool_t iny;
    // inside Y?
    if(HasVaryingY<trdTypeT>::value != TrdTypes::kNo) {
      // If Trd type is unknown don't bother with a runtime check, assume
      // the general case
      PointLineOrientation<Backend>(Abs(point.y()) - trd.dy1(), pzPlusDz, trd.y2minusy1(), trd.dztimes2(), cross);
      iny = cross >= 0;
    }
    else {
      iny = Abs(point.y()) <= trd.dy1();
    }

    inside = inz & inx & iny;
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

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(trd, localPoint, inside);
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

    Bool_t isInside;
    UnplacedContains<Backend>(trd, localpoint, isInside);
    MaskedAssign(isInside, EInside::kInside, &inside);
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
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;

    Float_t hitx, hity, hitz;

    Vector3D<Float_t> pos_local;
    Vector3D<Float_t> dir_local;
    distance = kInfinity;

    transformation.Transform<transCodeT, rotCodeT>(point, pos_local);
    transformation.TransformDirection<rotCodeT>(direction, dir_local);

    //Float_t abspx = Abs(pos_local.x());
    //Float_t abspy = Abs(pos_local.y());
    //Float_t abspz = Abs(pos_local.z());

    // hit Z faces?
    Float_t distz = (Abs(pos_local.z()) - trd.dz()) / Abs(dir_local.z());
    //std::cout << "distz candidate: " << distz << std::endl;
    // exclude case in which particle is going away
    Bool_t okz = pos_local.z() * dir_local.z() < 0;
      
    hitx = Abs(pos_local.x() + distz*dir_local.x());
    hity = Abs(pos_local.y() + distz*dir_local.y());

    // hitting top face?
    Bool_t okzt = pos_local.z() > trd.dz() && hitx <= trd.dx2() && hity <= trd.dy2();
    // hitting bottom face?
    Bool_t okzb = pos_local.z() < - trd.dz() && hitx <= trd.dx1() && hity <= trd.dy1();
     
    okz &= (okzt | okzb);
    MaskedAssign(okz, distz, &distance);

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

    FaceTrajectoryIntersection<Backend, true, false>(trd, pos_local, dir_local, disty, oky);
    MaskedAssign(oky && disty < distance, disty, &distance);

    FaceTrajectoryIntersection<Backend, true, true>(trd, pos_local, dir_local, disty, oky);
    MaskedAssign(oky && disty < distance, disty, &distance);
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
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;
    
    Float_t hitx, hity, hitz;
    Bool_t done = Backend::kFalse;
    distance = kInfinity;

    // hit top Z face?
    Float_t distz = (trd.dz() - point.z()) / Abs(dir.z());
    //std::cout << distz << std::endl;
    if(distz > 0) {
      hitx = Abs(point.x() + distz*dir.x());
      hity = Abs(point.y() + distz*dir.y());
      Bool_t okzt = dir.z() > 0;
      okzt &= hitx <= trd.dx2() && hity <= trd.dy2();
      MaskedAssign(okzt, distz, &distance);
      if(Backend::early_returns) {
          done |= okzt;
          if(done) return;
      }
      //std::cout << okzt << std::endl;
    }

    // hit bottom Z face?
    distz = (point.z() - (-trd.dz()) ) / Abs(dir.z());
    if(distz > 0) {
      hitx = Abs(point.x() + distz*dir.x());
      hity = Abs(point.y() + distz*dir.y());
      Bool_t okzt = dir.z() < 0;
      okzt &= hitx <= trd.dx1() && hity <= trd.dy1();
      MaskedAssign(okzt, distz, &distance);
      if(Backend::early_returns) {
          done |= okzt;
          if(done) return;
      }
      //std::cout << okzt << std::endl;
    }

    // hitting X faces?
    Float_t distx;
    Bool_t okx;

    FaceTrajectoryIntersection<Backend, false, false>(trd, point, dir, distx, okx);
    //std::cout << distx << std::endl;
    MaskedAssign(okx & (distx < distance), distx, &distance);
    if(Backend::early_returns) {
        done |= okx;
        if(done) return;
    }
      
    FaceTrajectoryIntersection<Backend, false, true>(trd, point, dir, distx, okx);
    MaskedAssign(okx & (distx < distance), distx, &distance);
    if(Backend::early_returns) {
        done |= okx;
        if(done) return;
    }
    //std::cout << distx << std::endl;

    // hitting X faces?
    Float_t disty;
    Bool_t oky;

    FaceTrajectoryIntersection<Backend, true, false>(trd, point, dir, disty, oky);
    //std::cout << disty << std::endl;
    MaskedAssign(oky & (disty < distance), disty, &distance);
    if(Backend::early_returns) {
        done |= oky;
        if(done) return;
    }
      
    FaceTrajectoryIntersection<Backend, true, true>(trd, point, dir, disty, oky);
    MaskedAssign(oky & (disty < distance), disty, &distance);
    //std::cout << disty << std::endl;
    if(Backend::early_returns) {
        done |= oky;
        if(done) return;
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
    Safety<Backend, false>(trd, pos_local, safety);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTrd const &trd,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {
    using namespace TrdUtilities;
    Safety<Backend, true>(trd, point, safety); 
  }

};

}

#endif // VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
