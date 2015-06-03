/// @file TubeImplementation.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTube.h"
#include "volumes/kernel/shapetypes/TubeTypes.h"
#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(TubeImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric, typename)

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace TubeUtilities {

/**
 * Returns whether a point is inside a cylindrical sector, as defined
 * by the two vectors that go along the endpoints of the sector
 *
 * The same could be achieved using atan2 to calculate the angle formed
 * by the point, the origin and the X-axes, but this is a lot faster,
 * using only multiplications and comparisons
 *
 * (-x*starty + y*startx) >= 0: calculates whether going from the start vector to the point
 * we are traveling in the CCW direction (taking the shortest direction, of course)
 *
 * (-endx*y + endy*x) >= 0: calculates whether going from the point to the end vector
 * we are traveling in the CCW direction (taking the shortest direction, of course)
 *
 * For a sector smaller than pi, we need that BOTH of them hold true - if going from start, to the
 * point, and then to the end we are travelling in CCW, it's obvious the point is inside the
 * cylindrical sector. 
 * 
 * For a sector bigger than pi, only one of the conditions needs to be true. This is less obvious why.
 * Since the sector angle is greater than pi, it can be that one of the two vectors might be
 * farther than pi away from the point. In that case, the shortest direction will be CW, so even
 * if the point is inside, only one of the two conditions need to hold.
 * 
 * If going from start to point is CCW, then certainly the point is inside as the sector
 * is larger than pi.
 *
 * If going from point to end is CCW, again, the point is certainly inside.
 *
 * This function is a frankensteinian creature that can determine which of the two cases (smaller vs larger than pi)
 * to use either at compile time (if it has enough information, saving an if statement) or at runtime.
**/

template<typename Backend, typename ShapeType, typename UnplacedVolumeType, bool onSurfaceT>
 VECGEOM_INLINE
 VECGEOM_CUDA_HEADER_BOTH
 void PointInCyclicalSector(UnplacedVolumeType const& volume, typename Backend::precision_v const &x, 
   typename Backend::precision_v const &y, typename Backend::bool_v &ret) {
  using namespace TubeTypes;
  // assert(SectorType<ShapeType>::value != kNoAngle && "ShapeType without a sector passed to PointInCyclicalSector");

  typedef typename Backend::precision_v Float_t;

  Float_t startx = volume.alongPhi1x();
  Float_t starty = volume.alongPhi1y();

  Float_t endx = volume.alongPhi2x();
  Float_t endy = volume.alongPhi2y();

  bool smallerthanpi;

  if(SectorType<ShapeType>::value == kUnknownAngle)
    smallerthanpi = volume.dphi() <= M_PI;
  else
    smallerthanpi = SectorType<ShapeType>::value == kOnePi || SectorType<ShapeType>::value == kSmallerThanPi;

  Float_t startCheck = (-x*starty + y*startx);
  Float_t endCheck = (-endx*y + endy*x);

  if(onSurfaceT) {
    ret = (Abs(startCheck) <= kTolerance) | (Abs(endCheck) <= kTolerance);
  }
  else {
    if(smallerthanpi) {
      ret = (startCheck >= -kTolerance) & (endCheck >= -kTolerance);
    }
    else {
      ret = (startCheck >= -kTolerance) | (endCheck >= -kTolerance);
    }    
  }
}

template<typename Backend, typename TubeType, bool LargestSolution, bool insectorCheck>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CircleTrajectoryIntersection(typename Backend::precision_v const &b,
                                  typename Backend::precision_v const &c,
                                  UnplacedTube const& tube,
                                  Vector3D<typename Backend::precision_v> const &pos,
                                  Vector3D<typename Backend::precision_v> const &dir,
                                  typename Backend::precision_v &dist,
                                  typename Backend::bool_v &ok) {
  using namespace TubeTypes;
  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t delta = b*b - c;
  Bool_t delta_mask;
  if(!LargestSolution)
   delta_mask = delta > 0.;
  else
   delta_mask = delta >= 0.;
  MaskedAssign(!delta_mask, 0. , &delta);
  delta = Sqrt(delta);

  if(!LargestSolution)
    delta = -delta;

  dist = -b + delta;

  if(insectorCheck) {
    Float_t hitz = pos.z() + dist * dir.z();
    Float_t hity = pos.y() + dist * dir.y();
    Float_t hitx = pos.x() + dist * dir.x();

    Bool_t insector = Backend::kTrue;
    if(checkPhiTreatment<TubeType>(tube)) {
//      PointInCyclicalSector<Backend, TubeType, UnplacedTube, false>(tube, hitx, hity, insector);
        insector = tube.GetWedge().ContainsWithBoundary<Backend>( Vector3D<typename Backend::precision_v>(hitx, hity, hitz) );
    }
    ok = delta_mask & (dist >= 0) & (Abs(hitz) <= tube.z()) & insector;
  }
  else {
    ok = delta_mask;
  }
}

/*
 * Input: A point p and a unit vector v.
 * Returns the perpendicular distance between
 * (the infinite line defined by the vector) 
 * and (the point)
 * 
 * How does it work? Let phi be the angle formed
 * by v and the position vector of the point.
 * 
 * Let proj be the projection vector of point onto that
 * line.
 * 
 * We now have a right triangle formed by points
 * (0, 0), point and the projected point
 * 
 * For that triangle, it holds that
 * sin theta = perpendiculardistance / (magnitude of point vector)
 * 
 * So perpendiculardistance = sin theta * (magnitude of point vector)
 *
 * But.. the magnitude of the cross product between the point vector
 * and the v vector is:
 * 
 * |p x v| = |p| * |v| * sin theta
 * 
 * Since |v| = 1, the magnitude of the cross product is exactly
 * what we're looking for, the formula for which is simply
 * p.x * v.y - p.y * v.x
 *
 */

template<typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PerpendicularDistance2D(typename Backend::precision_v const &px, typename Backend::precision_v const &py,
                            typename Backend::precision_v const &vx, typename Backend::precision_v const &vy,
                            typename Backend::precision_v &dist) {
  dist = px * vy - py * vx;
}

/*
 * Find safety distance from a point to the phi plane
 */

template<typename Backend, typename TubeType, bool inside>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PhiPlaneSafety( UnplacedTube const& tube,
                     Vector3D<typename Backend::precision_v> const &pos,
                     typename Backend::precision_v &safety) {
  using namespace TubeTypes;
  typedef typename Backend::precision_v Float_t;

  Float_t phi1;
  PerpendicularDistance2D<Backend>(pos.x(), pos.y(), tube.alongPhi1x(), tube.alongPhi1y(), phi1);
  if(inside)
    phi1 *= -1;

  if(SectorType<TubeType>::value == kOnePi) {
    safety = Abs(phi1);
    return;
  }

  // make sure point falls on positive part of projection
  MaskedAssign(phi1<0 || pos.x()*tube.alongPhi1x()+pos.y()*tube.alongPhi1y() < 0, kInfinity, &phi1);
  Float_t phi2;
  PerpendicularDistance2D<Backend>(pos.x(), pos.y(), tube.alongPhi2x(), tube.alongPhi2y(), phi2);
  if(inside)
    phi2 *= -1;

  // make sure point falls on positive part of projection
  phi2 *= -1;
  MaskedAssign(phi2<0 || pos.x()*tube.alongPhi2x()+pos.y()*tube.alongPhi2y() < 0, kInfinity, &phi2);
  safety = Min(phi1, phi2);
  MaskedAssign(safety == kInfinity, Float_t(-1), &safety);
}

/*
 * Check intersection of the trajectory with a phi-plane
 * All points of the along-vector of a phi plane lie on
 * s * (alongX, alongY)
 * All points of the trajectory of the particle lie on
 * (x, y) + t * (vx, vy)
 * Thefore, it must hold that s * (alongX, alongY) == (x, y) + t * (vx, vy)
 * Solving by t we get t = (alongY*x - alongX*y) / (vy*alongX - vx*alongY)
 * s = (x + t*vx) / alongX = (newx) / alongX
 *
 * If we have two non colinear phi-planes, need to make sure
 * point falls on its positive direction <=> dot product between
 * along vector and hit-point is positive <=> hitx*alongX + hity*alongY > 0
 */

template<typename Backend, typename TubeType, bool PositiveDirectionOfPhiVector, bool insectorCheck>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PhiPlaneTrajectoryIntersection(Precision alongX, Precision alongY,
                                    UnplacedTube const& tube,
                                    Vector3D<typename Backend::precision_v> const &pos,
                                    Vector3D<typename Backend::precision_v> const &dir,
                                    typename Backend::precision_v &dist,
                                    typename Backend::bool_v &ok) {

  typedef typename Backend::precision_v Float_t;

  dist = (alongY*pos.x() - alongX*pos.y() ) / (dir.y()*alongX - dir.x()*alongY);


  if(insectorCheck) {
    Float_t hitx = pos.x() + dist * dir.x();
    Float_t hity = pos.y() + dist * dir.y();
    Float_t hitz = pos.z() + dist * dir.z();
    Float_t r2 = hitx*hitx + hity*hity;

    ok = Abs(hitz) <= tube.tolIz() &&
          (r2 >= tube.tolIrmin2()) &&
          (r2 <= tube.tolIrmax2()) &&
          dist > 0;

    if(PositiveDirectionOfPhiVector)
      ok = ok && (hitx*alongX + hity*alongY) > 0.;
  }
  else {
    if(PositiveDirectionOfPhiVector) {
      Float_t hitx = pos.x() + dist * dir.x();
      Float_t hity = pos.y() + dist * dir.y();
      ok = (hitx*alongX + hity*alongY) >= 0.;
    }
  }
}
}

class PlacedTube;

template <TranslationCode transCodeT, RotationCode rotCodeT, typename tubeTypeT>
struct TubeImplementation {

#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL int transC = transCodeT;
  VECGEOM_GLOBAL int rotC   = rotCodeT;
#else
  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;
#endif

  using PlacedShape_t = PlacedTube;
  using UnplacedShape_t = UnplacedTube;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedTube<%i, %i, %s>", transCodeT, rotCodeT, tubeTypeT::toString());
  }


  /////GenericKernel Contains/Inside implementation
    template <typename Backend, bool ForInside>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void GenericKernelForContainsAndInside(UnplacedTube const &tube,
              Vector3D<typename Backend::precision_v> const &point,
              typename Backend::bool_v &completelyinside,
              typename Backend::bool_v &completelyoutside) {

        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;

        // very fast check on z-height
        Float_t absz = Abs(point[2]);
        completelyoutside = absz > MakePlusTolerant<ForInside>( tube.z() );
        if (ForInside)
        {
            completelyinside = absz < MakeMinusTolerant<ForInside>( tube.z() );
        }
        if (Backend::early_returns) {
            if ( IsFull(completelyoutside) ) {
              return;
            }
        }

        // check on RMAX
        Float_t r2 = point.x()*point.x()+point.y()*point.y();
        // calculate cone radius at the z-height of position

        completelyoutside |= r2 > MakePlusTolerantSquare<ForInside>( tube.rmax(), tube.rmax2() );
        if (ForInside)
        {
          completelyinside &= r2 < MakeMinusTolerantSquare<ForInside>( tube.rmax(), tube.rmax2() );
        }
        if (Backend::early_returns) {
                if ( IsFull(completelyoutside) ) {
                  return;
                }
        }

        // check on RMIN
        if (TubeTypes::checkRminTreatment<tubeTypeT>(tube)) {
          completelyoutside |= r2 <= MakeMinusTolerantSquare<ForInside>( tube.rmin(), tube.rmin2() );
          if (ForInside)
          {
           completelyinside &= r2 > MakePlusTolerantSquare<ForInside>( tube.rmin(), tube.rmin2() );
          }
            if (Backend::early_returns) {
               if ( IsFull(completelyoutside) ) {
                  return;
               }
            }
        }

        if(TubeTypes::checkPhiTreatment<tubeTypeT>(tube)) {
              Bool_t completelyoutsidephi;
              Bool_t completelyinsidephi;
              tube.GetWedge().GenericKernelForContainsAndInside<Backend,ForInside>( point,
                completelyinsidephi, completelyoutsidephi );

              completelyoutside |= completelyoutsidephi;
              if( ForInside )
                completelyinside &= completelyinsidephi;
           }
    }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedTube const &tube,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &contains) {

      typedef typename Backend::bool_v Bool_t;
      Bool_t unused;
      Bool_t outside;
      GenericKernelForContainsAndInside<Backend, false>(tube, point, unused, outside);
      contains = !outside;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedTube const &tube,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(tube, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTube const &tube,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside) {

      Vector3D<typename Backend::precision_v> localPoint
            = transformation.Transform<transCodeT, rotCodeT>(point);

      typedef typename Backend::bool_v Bool_t;
      Bool_t completelyinside, completelyoutside;
      GenericKernelForContainsAndInside<Backend,true>(tube,
          localPoint, completelyinside, completelyoutside);
      inside = EInside::kSurface;
      MaskedAssign(completelyoutside, EInside::kOutside, &inside);
      MaskedAssign(completelyinside,  EInside::kInside, &inside);
  }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void DistanceToIn(
        UnplacedTube const &tube,
        Transformation3D const &transformation,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &direction,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {
      
    using namespace TubeUtilities;
    using namespace TubeTypes;  
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Vector3D<Float_t> pos_local;
    Vector3D<Float_t> dir_local;
    distance = kInfinity;

    transformation.Transform<transCodeT, rotCodeT>(point, pos_local);
    transformation.TransformDirection<rotCodeT>(direction, dir_local);

    Float_t hitx, hity, r2;
    Float_t rsq = pos_local.x()*pos_local.x() + pos_local.y()*pos_local.y();
    Float_t pos_dot_dir_x = pos_local.x()*dir_local.x();
    Float_t pos_dot_dir_y = pos_local.y()*dir_local.y();


    // Determine which particles hit the Z face
    Float_t abspz = Abs(pos_local.z());
    Float_t distz = (abspz - tube.z()) / Abs( dir_local.z() );
    // exclude case in which particle is going away
    Bool_t okz = (pos_local.z() * dir_local.z()) < 0;

    // outside of Z range and going away?
    if(Backend::early_returns && abspz > tube.z() && !okz) return;
    // outside of tube and going away?
    if(Backend::early_returns && Abs(pos_local.x()) > tube.rmax() && pos_dot_dir_x >= 0) return;
    if(Backend::early_returns && Abs(pos_local.y()) > tube.rmax() && pos_dot_dir_y >= 0) return;



    hitx = pos_local.x() + distz*dir_local.x();
    hity = pos_local.y() + distz*dir_local.y();
    r2 = hitx*hitx + hity*hity;

    okz =  (okz) & (distz > 0) & (r2 <= tube.tolIrmax2());
    if(checkRminTreatment<tubeTypeT>(tube)) {
      okz = okz && (tube.tolIrmin2() <= r2);
    }
    if(checkPhiTreatment<tubeTypeT>(tube)) {
      Bool_t insector;
      PointInCyclicalSector<Backend, tubeTypeT, UnplacedTube, false>(tube, hitx, hity, insector);
      okz = okz & insector;
    }

    if(Backend::early_returns && okz == Backend::kTrue) {
      distance = distz;
      return;
    }

    /*
     * Calculate the intersection of the trajectory of
     * the particles with the two circles
     *
     * Here I compute values used in both rmin and
     * rmax calculations
     */

    Float_t invnsq = 1 / ( 1 - dir_local.z()*dir_local.z() );
    Float_t rdotn = pos_dot_dir_x + pos_dot_dir_y;
    Float_t b = invnsq * rdotn;

    /*
     * rmax
     * If the particle were to hit rmax, it would hit
     * the closest point of the two - I only consider
     * the smallest solution
     */
    Float_t crmax = invnsq * (rsq - tube.rmax2());
    Float_t dist_rmax;
    Bool_t ok_rmax;
    CircleTrajectoryIntersection<Backend, tubeTypeT, false, true>(b, crmax, tube, pos_local, dir_local, dist_rmax, ok_rmax);

    if(Backend::early_returns && ok_rmax == Backend::kTrue) {
      distance = dist_rmax;
      return;
    }
    
    /*
     * rmin
     * If the particle were to hit rmin, it would hit
     * the furthest away point of the two - so I only 
     * consider the largest solution to the quadratic 
     * equation
     */
    Float_t dist_rmin;
    Bool_t ok_rmin(false);
    if(checkRminTreatment<tubeTypeT>(tube)) {
      Float_t crmin = invnsq * (rsq - tube.rmin2());
      CircleTrajectoryIntersection<Backend, tubeTypeT, true, true>(b, crmin, tube, pos_local, dir_local, dist_rmin, ok_rmin);
    }

    /* 
     * What happens if both intersections are valid
     * for the same particle?
     *
     * This can only happen when particle is outside of 
     * the hollow space and will certainly hit rmax, not rmin
     *
     * So rmax solution always takes priority over rmin, and
     * will overwrite it in case both are valid
     */

    if(checkRminTreatment<tubeTypeT>(tube))
      MaskedAssign(ok_rmin, dist_rmin, &distance);
    MaskedAssign(ok_rmax, dist_rmax, &distance);

    /*
     * Now write result from hitting Z face
     */
    MaskedAssign(okz && distz < distance && distz >= 0., distz, &distance);

    /*
     * Calculate intersection between trajectory and the 
     * two phi planes
     */

    if(checkPhiTreatment<tubeTypeT>(tube)) {

      Float_t dist_phi;
      Bool_t ok_phi;

      PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, SectorType<tubeTypeT>::value != kOnePi, true>(
              tube.alongPhi1x(), tube.alongPhi1y(),
              tube, pos_local, dir_local, dist_phi, ok_phi);

      MaskedAssign(ok_phi && dist_phi < distance, dist_phi, &distance);

      /*
       * If the tube is pi degrees, there's just one phi plane,
       * so no need to check again
       */

      if(SectorType<tubeTypeT>::value != kOnePi) {
        PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, true, true>(tube.alongPhi2x(), tube.alongPhi2y(),
              tube, pos_local, dir_local, dist_phi, ok_phi);

        MaskedAssign(ok_phi && dist_phi < distance, dist_phi, &distance);
      }
    }

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTube const &tube,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    using namespace TubeTypes;
    using namespace TubeUtilities;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    distance = kInfinity;

    Float_t dirzsign(1.);
    MaskedAssign(dir.z() < 0, Float_t(-1.), &dirzsign);
    Float_t distz = (tube.z()*dirzsign - point.z()) / dir.z();
    MaskedAssign(distz >= 0, distz, &distance);

    /*
     * Calculate the intersection of the trajectory of
     * the particles with the two circles
     *
     * Here I compute values used in both rmin and
     * rmax calculations
     */

    Float_t invnsq = 1 / ( dir.x()*dir.x() + dir.y()*dir.y() );
    Float_t rdotn = dir.x()*point.x() + dir.y()*point.y();
    Float_t rsq = point.x()*point.x() + point.y()*point.y();
    Float_t b = invnsq * rdotn;

    /*
     * rmin
     */

    Float_t dist_rmin;
    Bool_t ok_rmin;
    if(checkRminTreatment<tubeTypeT>(tube)) {
      Float_t crmin = invnsq * (rsq - tube.rmin2());
      CircleTrajectoryIntersection<Backend, tubeTypeT, false, false>(b, crmin, tube, point, dir, dist_rmin, ok_rmin);
      MaskedAssign(ok_rmin && dist_rmin >= 0 && dist_rmin < distance, dist_rmin, &distance);
    }

    /*
     * rmax
     */

    Float_t dist_rmax;
    Bool_t ok_rmax;
    Float_t crmax = invnsq * (rsq - tube.rmax2());
    CircleTrajectoryIntersection<Backend, tubeTypeT, true, false>(b, crmax, tube, point, dir, dist_rmax, ok_rmax);
    MaskedAssign(ok_rmax && dist_rmax >= 0 && dist_rmax < distance, dist_rmax, &distance);

    /* Phi planes 
     *
     * OK, this is getting weird - the only time I need to
     * check if hit-point falls on the positive direction
     * of the phi-vector is when angle is bigger than PI.
     *
     * Otherwise, any distance I get from there is guaranteed to
     * be larger - so final result would still be correct and no need to
     * check it
     */

    if(checkPhiTreatment<tubeTypeT>(tube)) {
      Float_t dist_phi;
      Bool_t ok_phi;
      Bool_t unused;

      if(SectorType<tubeTypeT>::value == kSmallerThanPi) { 
        PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, false, false>(
            tube.alongPhi1x(), tube.alongPhi1y(), tube, point, dir, dist_phi, unused);
        MaskedAssign(dist_phi > 0 && dist_phi < distance, dist_phi, &distance);

        PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, false, false>(
          tube.alongPhi2x(), tube.alongPhi2y(), tube, point, dir, dist_phi, unused);
        MaskedAssign(dist_phi > 0 && dist_phi < distance, dist_phi, &distance);
      }
      else if(SectorType<tubeTypeT>::value == kOnePi) {
        PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, false, false>(
          tube.alongPhi2x(), tube.alongPhi2y(), tube, point, dir, dist_phi, unused);
        MaskedAssign(dist_phi > 0 && dist_phi < distance, dist_phi, &distance);
      }
      else {
        // angle bigger than pi or unknown
        // need to check that point falls on positive direction of phi-vectors
        PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, true, false>(
            tube.alongPhi1x(), tube.alongPhi1y(), tube, point, dir, dist_phi, ok_phi);
        MaskedAssign(ok_phi && dist_phi > 0 && dist_phi < distance, dist_phi, &distance);

        PhiPlaneTrajectoryIntersection<Backend, tubeTypeT, true, false>(
          tube.alongPhi2x(), tube.alongPhi2y(), tube, point, dir, dist_phi, ok_phi);
        MaskedAssign(ok_phi && dist_phi > 0 && dist_phi < distance, dist_phi, &distance);
      }
    }

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTube const &tube,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {


    using namespace TubeTypes;
    using namespace TubeUtilities;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Vector3D<Float_t> local_point = transformation.Transform<transCodeT,rotCodeT>(point);
    safety = 0;

    Float_t r = Sqrt(local_point.x()*local_point.x() + local_point.y()*local_point.y());

    Float_t safez = Abs(local_point.z()) - tube.z();
    Float_t safermax = r - tube.rmax();

    safety = Max(safez, safermax);

    if(checkRminTreatment<tubeTypeT>(tube)) {
      Float_t safermin = tube.rmin() - r;
      safety = Max(safety, safermin);
    }

    if(checkPhiTreatment<tubeTypeT>(tube)) {

      Bool_t insector;
      PointInCyclicalSector<Backend, tubeTypeT, UnplacedTube, false>(tube, local_point.x(), local_point.y(), insector);

      if(Backend::early_returns && insector) return;

      Float_t safephi;
      PhiPlaneSafety<Backend, tubeTypeT, false>(tube, local_point, safephi);
      MaskedAssign(insector, Float_t(-1), &safephi);
      safety = Max(safety, safephi);
    }



  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTube const &tube,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {

     using namespace TubeTypes;
     using namespace TubeUtilities;
     typedef typename Backend::precision_v Float_t;

     safety = 0;
     Float_t r = Sqrt(point.x()*point.x() + point.y()*point.y());
     Float_t safez = tube.z() - Abs(point.z());
     Float_t safermax = tube.rmax() - r;

     safety = Min(safez, safermax);

     if(checkRminTreatment<tubeTypeT>(tube)) {
       Float_t safermin = r - tube.rmin();
       safety = Min(safety, safermin);
     }

    if(checkPhiTreatment<tubeTypeT>(tube)) {

      Float_t safephi;
      PhiPlaneSafety<Backend, tubeTypeT, true>(tube, point, safephi);
      MaskedAssign(safephi < 0, kInfinity, &safephi);
      safety = Min(safety, safephi);
    }
  }

}; // End of TubeUtilties

} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_
