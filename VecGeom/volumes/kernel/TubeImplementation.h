/// @file TubeImplementation.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/shapetypes/TubeTypes.h"
#include "VecGeom/volumes/TubeStruct.h"
#include "VecGeom/volumes/Wedge.h"
#include <cstdio>

#define TUBE_SAFETY_OLD // use old (and faster) definitions of SafetyToIn() and
                        // SafetyToOut()

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, TubeImplementation, typename);

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
 * This function is a frankensteinian creature that can determine which of the two cases (smaller vs
 * larger than pi) to use either at compile time (if it has enough information, saving an if
 * statement) or at runtime.
 **/

template <typename Real_v, typename ShapeType, typename UnplacedVolumeType, bool onSurfaceT, bool includeSurface = true>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PointInCyclicalSector(UnplacedVolumeType const &volume,
                                                                        Real_v const &x, Real_v const &y,
                                                                        typename vecCore::Mask_v<Real_v> &ret)
{
  using namespace ::vecgeom::TubeTypes;
  // assert(SectorType<ShapeType>::value != kNoAngle && "ShapeType without a
  // sector passed to PointInCyclicalSector");

  Real_v startx(volume.fAlongPhi1x);
  Real_v starty(volume.fAlongPhi1y);

  Real_v endx(volume.fAlongPhi2x);
  Real_v endy(volume.fAlongPhi2y);

  bool smallerthanpi;

  if (SectorType<ShapeType>::value == kUnknownAngle)
    smallerthanpi = volume.fDphi <= M_PI;
  else
    smallerthanpi = SectorType<ShapeType>::value == kOnePi || SectorType<ShapeType>::value == kSmallerThanPi;

  Real_v startCheck = (-x * starty + y * startx);
  Real_v endCheck   = (-endx * y + endy * x);

  if (onSurfaceT) {
    // in this case, includeSurface is irrelevant
    ret = (Abs(startCheck) <= kHalfTolerance) || (Abs(endCheck) <= kHalfTolerance);
  } else {
    if (smallerthanpi) {
      if (includeSurface)
        ret = (startCheck >= -kHalfTolerance) & (endCheck >= -kHalfTolerance);
      else
        ret = (startCheck >= kHalfTolerance) & (endCheck >= kHalfTolerance);
    } else {
      if (includeSurface)
        ret = (startCheck >= -kHalfTolerance) || (endCheck >= -kHalfTolerance);
      else
        ret = (startCheck >= kHalfTolerance) || (endCheck >= kHalfTolerance);
    }
  }
}

template <typename Real_v, typename UnplacedStruct_t, typename TubeType, bool LargestSolution, bool insectorCheck>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CircleTrajectoryIntersection(
    Real_v const &b, Real_v const &c, UnplacedStruct_t const &tube, Vector3D<Real_v> const &pos,
    Vector3D<Real_v> const &dir, Real_v &dist, typename vecCore::Mask_v<Real_v> &ok)
{
  using namespace ::vecgeom::TubeTypes;

  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v delta = b * b - c;
  ok           = delta > Real_v(0.);
  if (LargestSolution) ok |= delta == Real_v(0.); // this takes care of scratching conventions

  vecCore::MaskedAssign(delta, !ok, Real_v(0.));
  delta = Sqrt(delta);
  if (!LargestSolution) delta = -delta;

  dist = -b + delta;
  // ok &= vecCore::math::Abs(dist) <= kTolerance;
  // vecCore::MaskedAssign(dist,ok,Real_v(0.));
  // A.G There may be points propagated to Rmax+tolerance which get here and NEED to se a valid negative crossing
  // at distance > tolerance, so we need to enlarge the tolerance
  ok &= dist >= -2 * kTolerance;
  if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(ok)) return;

  if (insectorCheck) {
    /* if dist > 100*tube.fRmax, then instead of solving quadratic again,
    ** use Newton method to recalculate the root, taking previous distance
    ** as initial guess for newton method.
    **
    ** Commenting the code for root recalculation, coz  that sometimes overfits
    ** and ShapeTester starts complaining for TestAccuracyDistanceToIn tests
    ** The condition is handled in DistanceToIn itself.
    **
    ** Still keeping the code in comments for reference.
    **
    ** Real_v x(0.), y(0.);
    ** vecCore::MaskedAssign(x, dist > 100 * tube.fRmax, pos.x() + dist * dir.x());
    ** vecCore::MaskedAssign(y, dist > 100 * tube.fRmax, pos.y() + dist * dir.y());
    ** vecCore::MaskedAssign(dist, dist > 100 * tube.fRmax,
    **      dist - (x * x + y * y - tube.fRmax2) * 0.5 / NonZero(dir.x() * x + dir.y() * y));
    */

    Real_v hitz = pos.z() + dist * dir.z();
    ok &= (Abs(hitz) <= tube.fZ);
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(ok)) return;

    if (checkPhiTreatment<TubeType>(tube)) {
      Bool_v insector(false);
      Real_v hitx = pos.x() + dist * dir.x();
      Real_v hity = pos.y() + dist * dir.y();
      PointInCyclicalSector<Real_v, TubeType, UnplacedStruct_t, false, true>(tube, hitx, hity, insector);
      // insector = tube.fPhiWedge.ContainsWithBoundary<Real_v>(
      // Vector3D<Real_v>(hitx, hity, hitz) );
      ok &= insector;
    }
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

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v PerpDist2D(Real_v const &px, Real_v const &py, Real_v const &vx,
                                                               Real_v const &vy)
{
  return px * vy - py * vx;
}

/*
 * Find safety distance from a point to the phi plane
 */
template <typename Real_v, typename UnplacedStruct_t, typename TubeType, bool inside>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PhiPlaneSafety(UnplacedStruct_t const &tube,
                                                                 Vector3D<Real_v> const &pos, Real_v &safety)
{
  using namespace ::vecgeom::TubeTypes;

  if ((SectorType<TubeType>::value == kUnknownAngle && tube.fDphi > M_PI) ||
      (SectorType<TubeType>::value == kBiggerThanPi)) {
    safety = Sqrt(pos.x() * pos.x() + pos.y() * pos.y());
  } else {
    safety = kInfLength;
  }

  Real_v phi1 = PerpDist2D<Real_v>(pos.x(), pos.y(), Real_v(tube.fAlongPhi1x), Real_v(tube.fAlongPhi1y));
  if (inside) phi1 *= -1;

  if (SectorType<TubeType>::value == kOnePi) {
    auto absphi1 = Abs(phi1);
    vecCore::MaskedAssign(safety, absphi1 > kHalfTolerance, absphi1);
    return;
  }

  // make sure point falls on positive part of projection
  vecCore::MaskedAssign(safety,
                        phi1 > -kHalfTolerance &&
                            /*pos.x() * tube.fAlongPhi1x + pos.y() * tube.fAlongPhi1y > 0. &&*/ phi1 < safety,
                        phi1);

  Real_v phi2 = PerpDist2D<Real_v>(pos.x(), pos.y(), Real_v(tube.fAlongPhi2x), Real_v(tube.fAlongPhi2y));
  if (!inside) phi2 *= -1;

  // make sure point falls on positive part of projection
  vecCore::MaskedAssign(safety,
                        phi2 > -kHalfTolerance &&
                            /*pos.x() * tube.fAlongPhi2x + pos.y() * tube.fAlongPhi2y > 0. &&*/ phi2 < safety,
                        phi2);
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

template <typename Real_v, typename UnplacedStruct_t, typename TubeType, bool PositiveDirectionOfPhiVector,
          bool insectorCheck>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PhiPlaneTrajectoryIntersection(
    Precision alongX, Precision alongY, Precision normalX, Precision normalY, UnplacedStruct_t const &tube,
    Vector3D<Real_v> const &pos, Vector3D<Real_v> const &dir, Real_v &dist, typename vecCore::Mask_v<Real_v> &ok)
{

  dist = kInfLength;

  // approaching phi plane from the right side?
  // this depends whether we use it for DistanceToIn or DistanceToOut
  // Note: wedge normals poing towards the wedge inside, by convention!
  Real_v dirDotNorm = dir.x() * normalX + dir.y() * normalY;
  if (insectorCheck)
    ok = (dirDotNorm > Real_v(0.)); // DistToIn  -- require tracks entering volume
  else
    ok = (dirDotNorm < Real_v(0.)); // DistToOut -- require tracks leaving volume

  // if( vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(ok) ) return;

  Real_v dirDotXY = (dir.y() * alongX - dir.x() * alongY);
  dist            = (alongY * pos.x() - alongX * pos.y()) / NonZero(dirDotXY);
  // A.G to check validity, we have to compare with tolerance the safety rather than the distance to plane
  ok &= (dist * Abs(dirDotNorm)) > -kHalfTolerance;
  // if( vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(ok) ) return;

  if (insectorCheck) {
    Real_v hitx = pos.x() + dist * dir.x();
    Real_v hity = pos.y() + dist * dir.y();
    Real_v hitz = pos.z() + dist * dir.z();
    Real_v r2   = hitx * hitx + hity * hity;
    ok &= Abs(hitz) <= tube.fTolOz && (r2 >= tube.fTolOrmin2) && (r2 <= tube.fTolOrmax2);

    // GL: tested with this if(PosDirPhiVec) around if(insector), so
    // if(insector){} requires PosDirPhiVec==true to run
    //  --> shapeTester still finishes OK (no mismatches) (some cycles saved...)
    if (PositiveDirectionOfPhiVector) {
      ok = ok && (hitx * alongX + hity * alongY) > Real_v(0.);
    }
  } else {
    if (PositiveDirectionOfPhiVector) {
      Real_v hitx = pos.x() + dist * dir.x();
      Real_v hity = pos.y() + dist * dir.y();
      ok          = ok && (hitx * alongX + hity * alongY) >= Real_v(0.);
    }
  }
}

template <typename Real_v, typename UnplacedStruct_t, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsOnTubeSurface(
    UnplacedStruct_t const &tube, Vector3D<Real_v> const &point)
{
  const Real_v rho = point.Perp2();
  if (ForInnerSurface) {
    return (rho >= tube.fTolOrmin2) && (rho <= tube.fTolIrmin2) && (Abs(point.z()) < (tube.fZ + kTolerance));
  } else {
    return (rho >= tube.fTolIrmax2) && (rho <= tube.fTolOrmax2) && (Abs(point.z()) < (tube.fZ + kTolerance));
  }
}

template <typename Real_v, bool ForInnerSurface>
VECCORE_ATT_HOST_DEVICE Vector3D<Real_v> GetNormal(Vector3D<Real_v> const &point)
{
  Vector3D<Real_v> norm(0., 0., 0.);
  if (ForInnerSurface) {
    norm.Set(-point.x(), -point.y(), 0.);
  } else {
    norm.Set(point.x(), point.y(), 0.);
  }
  return norm;
}

template <typename Real_v, typename UnplacedStruct_t, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsMovingInsideTubeSurface(
    UnplacedStruct_t const &tube, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction)
{
  return IsOnTubeSurface<Real_v, UnplacedStruct_t, ForInnerSurface>(tube, point) &&
         (direction.Dot(GetNormal<Real_v, ForInnerSurface>(point)) <= Real_v(0.));
}

} // namespace TubeUtilities

template <typename T>
class SPlacedTube;
template <typename T>
class SUnplacedTube;
template <typename tubeTypeT>
struct TubeImplementation {

  using UnplacedStruct_t = ::vecgeom::TubeStruct<Precision>;
  using UnplacedVolume_t = SUnplacedTube<tubeTypeT>;
  using PlacedShape_t    = SPlacedTube<UnplacedVolume_t>;

  /////GenericKernel Contains/Inside implementation
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &tube, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &completelyinside,
      typename vecCore::Mask_v<Real_v> &completelyoutside)
  {
    using namespace ::vecgeom::TubeTypes;
    using Bool_v = vecCore::Mask_v<Real_v>;

    // very fast check on z-height
    Real_v absz       = Abs(point[2]);
    completelyoutside = absz > MakePlusTolerant<ForInside>(tube.fZ);
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<ForInside>(tube.fZ);
    }
    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    // check on RMAX
    Real_v r2 = point.x() * point.x() + point.y() * point.y();
    // calculate cone radius at the z-height of position

    completelyoutside |= r2 > MakePlusTolerantSquare<ForInside>(tube.fRmax);
    if (ForInside) {
      completelyinside &= r2 < MakeMinusTolerantSquare<ForInside>(tube.fRmax);
    }
    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    // check on RMIN
    if (checkRminTreatment<tubeTypeT>(tube)) {
      completelyoutside |= r2 <= MakeMinusTolerantSquare<ForInside>(tube.fRmin);
      if (ForInside) {
        completelyinside &= r2 > MakePlusTolerantSquare<ForInside>(tube.fRmin);
      }
      if (vecCore::EarlyReturnAllowed()) {
        if (vecCore::MaskFull(completelyoutside)) {
          return;
        }
      }
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      Bool_v completelyoutsidephi(false);
      Bool_v completelyinsidephi(false);
      tube.fPhiWedge.GenericKernelForContainsAndInside<Real_v, ForInside>(point, completelyinsidephi,
                                                                          completelyoutsidephi);

      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &tube,
                                                                    Vector3D<Real_v> const &point,
                                                                    typename vecCore::Mask_v<Real_v> &contains)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v unused, outside;
    GenericKernelForContainsAndInside<Real_v, false>(tube, point, unused, outside);
    contains = !outside;
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &tube,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, true>(tube, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &tube,
                                                                        Vector3D<Real_v> const &pointt,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    Vector3D<Real_v> point = pointt;
    Real_v ptDist          = point.Mag();
    Real_v distToMove(0.);
    using Bool_v    = vecCore::Mask_v<Real_v>;
    Precision order = 100.;
    Bool_v cond     = (ptDist > order * tube.fMaxVal);
    /* if the point is at a distance (DIST) of more than 100 times of the maximum dimension
     * (of the shape) from the origin of shape, then before calculating distance, first
     * manually move the point with distance ( distToMove = DIST-100.*maxDim) along the
     * direction, and then calculate DistanceToIn of new moved point using DistanceToInKernel,
     *
     * The final distance will be (distToMove + DistanceToIn),
     *
     * This logic no longer requires the recalculation of the roots using Newton method in
     * CircleTrajectoryIntersection function, and will also give consistent results with
     * ShapeTester
     */
    vecCore__MaskedAssignFunc(distToMove, cond, (ptDist - Real_v(order * tube.fMaxVal)));
    vecCore__MaskedAssignFunc(point, cond, point + distToMove * dir);
    DistanceToInKernel<Real_v>(tube, point, dir, stepMax, distance);
    distance += distToMove;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToInKernel(UnplacedStruct_t const &tube,
                                                                              Vector3D<Real_v> const &point,
                                                                              Vector3D<Real_v> const &dir,
                                                                              Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using namespace TubeUtilities;
    using namespace ::vecgeom::TubeTypes;

    using Bool_v = vecCore::Mask_v<Real_v>;

    Bool_v done(false);

    //=== First, for points outside and moving away --> return infinity
    distance = kInfLength;

    // outside of Z range and going away?
    Real_v distz = Abs(point.z()) - tube.fZ; // avoid a division for now
    done |= distz > kHalfTolerance && point.z() * dir.z() >= 0;

    // // outside of tube and going away?
    // done |= Abs(point.x()) > tube.rmax()+kHalfTolerance && point.x()*dir.x()
    // >= 0;
    // done |= Abs(point.y()) > tube.rmax()+kHalfTolerance && point.y()*dir.y()
    // >= 0;
    // if(vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    // outside of outer tube and going away?
    Real_v rsq   = point.x() * point.x() + point.y() * point.y();
    Real_v rdotn = point.x() * dir.x() + point.y() * dir.y();
    done |= rsq > tube.fTolIrmax2 && rdotn >= 0;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    //=== Next, check all dimensions of the tube, whether points are inside -->
    // return -1
    vecCore__MaskedAssignFunc(distance, !done, Real_v(-1.0));

    // For points inside z-range, return -1
    Bool_v inside = distz < -kHalfTolerance;

    inside &= rsq < tube.fTolIrmax2;
    if (checkRminTreatment<tubeTypeT>(tube)) {
      inside &= rsq > tube.fTolIrmin2;
    }
    if (checkPhiTreatment<tubeTypeT>(tube) && !vecCore::MaskEmpty(inside)) {
      Bool_v insector;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false, false>(tube, point.x(), point.y(), insector);
      inside &= insector;
      // inside &= tube.fPhiWedge.ContainsWithoutBoundary<Real_v>( point );  //
      // slower than PointInCyclicalSector()
    }
    done |= inside;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    //=== Next step: check if z-plane is the right entry point (both r,phi
    // should be valid at z-plane crossing)
    vecCore::MaskedAssign(distance, !done, Real_v(kInfLength));

    distz /= NonZeroAbs(dir.z());
    // std::cerr << "Dist : " << distz << std::endl;

    Real_v hitx = point.x() + distz * dir.x();
    Real_v hity = point.y() + distz * dir.y();
    Real_v r2   = hitx * hitx + hity * hity; // radius of intersection with z-plane
    Bool_v okz  = distz > -kHalfTolerance && (point.z() * dir.z() < 0);

    okz &= (r2 <= tube.fRmax2);
    if (checkRminTreatment<tubeTypeT>(tube)) {
      okz &= (tube.fRmin2 <= r2);
    }
    if (checkPhiTreatment<tubeTypeT>(tube) && !vecCore::MaskEmpty(okz)) {
      Bool_v insector;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false>(tube, hitx, hity, insector);
      okz &= insector;
      // okz &= tube.fPhiWedge.ContainsWithBoundary<Real_v>(
      // Vector3D<Real_v>(hitx, hity, 0.0) );
    }
    vecCore::MaskedAssign(distance, !done && okz, distz);
    done |= okz;

    Bool_v isOnSurfaceAndMovingInside = IsMovingInsideTubeSurface<Real_v, UnplacedStruct_t, false>(tube, point, dir);
    if (checkRminTreatment<tubeTypeT>(tube)) {
      isOnSurfaceAndMovingInside |= IsMovingInsideTubeSurface<Real_v, UnplacedStruct_t, true>(tube, point, dir);
    }

    if (!checkPhiTreatment<tubeTypeT>(tube)) {
      vecCore__MaskedAssignFunc(distance, !done && isOnSurfaceAndMovingInside, Real_v(0.));
      done |= isOnSurfaceAndMovingInside;
      if (vecCore::MaskFull(done)) return;
    } else {
      Bool_v insector(false);
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false>(tube, point.x(), point.y(), insector);
      vecCore__MaskedAssignFunc(distance, !done && insector && isOnSurfaceAndMovingInside, Real_v(0.));
      done |= (insector && isOnSurfaceAndMovingInside);
      if (vecCore::MaskFull(done)) return;
    }

    // std::cerr << "distance : " << distance << std::endl;
    // if(vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done) ) return;

    //=== Next step: intersection of the trajectories with the two circles

    // Here for values used in both rmin and rmax calculations
    Real_v invnsq = Real_v(1.) / NonZero(Real_v(1.) - dir.z() * dir.z());
    Real_v b      = invnsq * rdotn;

    /*
     * rmax
     * If the particle were to hit rmax, it would hit the closest point of the
     * two
     * --> only consider the smallest solution of the quadratic equation
     */
    Real_v crmax = invnsq * (rsq - tube.fRmax2);
    Real_v dist_rmax;
    Bool_v ok_rmax(false);
    CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, true>(b, crmax, tube, point, dir,
                                                                                   dist_rmax, ok_rmax);
    ok_rmax &= dist_rmax < distance;
    vecCore::MaskedAssign(distance, !done && ok_rmax, dist_rmax);
    done |= ok_rmax;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    /*
     * rmin
     * If the particle were to hit rmin, it would hit the farthest point of the
     * two
     * --> only consider the largest solution to the quadratic equation
     */
    Real_v dist_rmin(-kInfLength);
    Bool_v ok_rmin(false);
    if (checkRminTreatment<tubeTypeT>(tube)) {
      /*
       * What happens if both intersections are valid for the same particle?
       * This can only happen when particle is outside of the hollow space and
       * will certainly hit rmax, not rmin
       * So rmax solution always takes priority over rmin, and will overwrite it
       * in case both are valid
       */
      Real_v crmin = invnsq * (rsq - tube.fRmin2);
      CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, true>(b, crmin, tube, point, dir,
                                                                                    dist_rmin, ok_rmin);
      ok_rmin &= dist_rmin < distance;
      vecCore::MaskedAssign(distance, !done && ok_rmin, dist_rmin);
      // done |= ok_rmin; // can't be done here, it's wrong in case
      // phi-treatment is needed!
    }

    /*
     * Calculate intersection between trajectory and the two phi planes
     */
    if (checkPhiTreatment<tubeTypeT>(tube)) {

      Real_v dist_phi;
      Bool_v ok_phi;
      auto const &w = tube.fPhiWedge;
      PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, SectorType<tubeTypeT>::value != kOnePi, true>(
          tube.fAlongPhi1x, tube.fAlongPhi1y, w.GetNormal1().x(), w.GetNormal1().y(), tube, point, dir, dist_phi,
          ok_phi);
      ok_phi &= dist_phi < distance;
      vecCore::MaskedAssign(distance, !done && ok_phi, dist_phi);
      done |= ok_phi;
      //      if(vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done))
      //      return;

      /*
       * If the tube is pi degrees, there's just one phi plane,
       * so no need to check again
       */

      if (SectorType<tubeTypeT>::value != kOnePi) {
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, true>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
      }
    }
  } // end of DistanceToIn()

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &tube,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using namespace ::vecgeom::TubeTypes;
    using namespace TubeUtilities;

    using Bool_v = vecCore::Mask_v<Real_v>;

    distance = Real_v(-1.);
    Bool_v done(false);

    //=== First we check all dimensions of the tube, whether points are outside
    //--> return -1

    // For points outside z-range, return -1
    Real_v distz = tube.fZ - Abs(point.z()); // avoid a division for now
    done |= distz < -kHalfTolerance;         // distance is already set to -1
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    Real_v rsq   = point.x() * point.x() + point.y() * point.y();
    Real_v rdotn = dir.x() * point.x() + dir.y() * point.y();
    Real_v crmax = rsq - tube.fRmax2; // avoid a division for now
    Real_v crmin = rsq;

    // if outside of Rmax, return -1
    done |= crmax > Real_v(2.0 * kTolerance * tube.fRmax);
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    if (checkRminTreatment<tubeTypeT>(tube)) {
      // if point is within inner-hole of a hollow tube, it is outside of the
      // tube --> return -1
      crmin -= tube.fRmin2; // avoid a division for now
      done |= crmin < Real_v(-2.0 * kTolerance * tube.fRmin);
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;
    }

    // TODO: add outside check for phi-sections here
    if (checkPhiTreatment<tubeTypeT>(tube)) {
      Bool_v completelyoutsidephi(false);
      Bool_v completelyinsidephi(false);
      tube.fPhiWedge.GenericKernelForContainsAndInside<Real_v, true>(point, completelyinsidephi, completelyoutsidephi);

      done |= completelyoutsidephi;
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;
    }
    // OK, since we're here, then distance must be non-negative, and the
    // smallest of possible intersections
    vecCore::MaskedAssign(distance, !done, Real_v(kInfLength));

    Real_v invdirz = Real_v(1.) / NonZero(dir.z());
    distz          = (tube.fZ - point.z()) * invdirz;
    vecCore__MaskedAssignFunc(distz, dir.z() < 0, (-tube.fZ - point.z()) * invdirz);
    vecCore::MaskedAssign(distance, !done && dir.z() != Real_v(0.) && distz < distance, distz);

    /*
     * Find the intersection of the trajectories with the two circles.
     * Here I compute values used in both rmin and rmax calculations.
     */

    Real_v invnsq = Real_v(1.) / NonZero(Real_v(1.) - dir.z() * dir.z());
    Real_v b      = invnsq * rdotn;

    /*
     * rmin
     */

    if (checkRminTreatment<tubeTypeT>(tube)) {
      Real_v dist_rmin(kInfLength);
      Bool_v ok_rmin(false);
      crmin *= invnsq;
      CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(b, crmin, tube, point, dir,
                                                                                      dist_rmin, ok_rmin);
      vecCore::MaskedAssign(distance, ok_rmin && dist_rmin < distance, dist_rmin);
    }

    /*
     * rmax
     */

    Real_v dist_rmax(kInfLength);
    Bool_v ok_rmax(false);
    crmax *= invnsq;
    CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, false>(b, crmax, tube, point, dir,
                                                                                   dist_rmax, ok_rmax);
    vecCore::MaskedAssign(distance, ok_rmax && dist_rmax < distance, dist_rmax);

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

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      Real_v dist_phi(kInfLength);
      Bool_v ok_phi(false);

      auto const &w = tube.fPhiWedge;
      if (SectorType<tubeTypeT>::value == kSmallerThanPi) {

        Precision normal1X = w.GetNormal1().x();
        Precision normal1Y = w.GetNormal1().y();
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(
            tube.fAlongPhi1x, tube.fAlongPhi1y, normal1X, normal1Y, tube, point, dir, dist_phi, ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);

        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
      } else if (SectorType<tubeTypeT>::value == kOnePi) {
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().x(), tube, point, dir, dist_phi,
            ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
      } else {
        // angle bigger than pi or unknown
        // need to check that point falls on positive direction of phi-vectors
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, false>(
            tube.fAlongPhi1x, tube.fAlongPhi1y, w.GetNormal1().x(), w.GetNormal1().y(), tube, point, dir, dist_phi,
            ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);

        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, false>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
      }
    }
    return;
  }

  /// This function keeps track of both positive (outside) and negative (inside)
  /// distances separately
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyAssign(Real_v safety, Real_v &positiveSafety,
                                                                        Real_v &negativeSafety)
  {
    vecCore::MaskedAssign(positiveSafety, safety >= Real_v(0.) && safety < positiveSafety, safety);
    vecCore::MaskedAssign(negativeSafety, safety <= Real_v(0.) && safety > negativeSafety, safety);
  }

  /** SafetyKernel finds distances from point to each face of the tube,
     returning
      largest negative distance (w.r.t. faces which point is inside of ) and
      smallest positive distance (w.r.t. faces which point is outside of)
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyKernel(UnplacedStruct_t const &tube,
                                                                        Vector3D<Real_v> const &point, Real_v &safePos,
                                                                        Real_v &safeNeg)
  {

    // TODO: implement caching if input point is not changed
    using namespace ::vecgeom::TubeTypes;
    using namespace TubeUtilities;

    safePos = kInfLength;
    safeNeg = -safePos; // reuse to avoid casting overhead

    Real_v safez = Abs(point.z()) - tube.fZ;
    SafetyAssign(safez, safePos, safeNeg);

    Real_v r        = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v safermax = r - tube.fRmax;
    SafetyAssign(safermax, safePos, safeNeg);

    if (checkRminTreatment<tubeTypeT>(tube)) {
      Real_v safermin = tube.fRmin - r;
      SafetyAssign(safermin, safePos, safeNeg);
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      Real_v safephi;
      PhiPlaneSafety<Real_v, UnplacedStruct_t, tubeTypeT, false>(tube, point, safephi);
      SafetyAssign(safephi, safePos, safeNeg);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &tube,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {

#ifdef TUBE_SAFETY_OLD
    SafetyToInOld(tube, point, safety);
#else
    Real_v safetyInsidePoint, safetyOutsidePoint;
    SafetyKernel(tube, point, safetyOutsidePoint, safetyInsidePoint);

    // Mostly called for points outside --> safetyOutside is finite --> return
    // safetyOutside
    // If safetyOutside == infinity --> return safetyInside
    safety = vecCore::Blend(safetyOutsidePoint == InfinityLength<Real_v>(), safetyInsidePoint, safetyOutsidePoint);
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &tube,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
#ifdef TUBE_SAFETY_OLD
    SafetyToOutOld(tube, point, safety);
#else
    Real_v safetyInsidePoint, safetyOutsidePoint;
    SafetyKernel<Real_v>(tube, point, safetyOutsidePoint, safetyInsidePoint);

    // Mostly called for points inside --> safetyOutside==infinity, return
    // |safetyInside| (flip sign)
    // If called for points outside -- return -safetyOutside
    safety = -vecCore::Blend(safetyOutsidePoint == InfinityLength<Real_v>(), safetyInsidePoint, safetyOutsidePoint);
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToInOld(UnplacedStruct_t const &tube,
                                                                         Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace ::vecgeom::TubeTypes;
    using namespace TubeUtilities;

    using Bool_v = vecCore::Mask_v<Real_v>;

    safety = Abs(point.z()) - tube.fZ;

    Real_v r        = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v safermax = r - tube.fRmax;
    vecCore::MaskedAssign(safety, safermax > safety, safermax);

    if (checkRminTreatment<tubeTypeT>(tube)) {
      Real_v safermin = tube.fRmin - r;
      vecCore::MaskedAssign(safety, safermin > safety, safermin);
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      Bool_v insector;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false, false>(tube, point.x(), point.y(), insector);
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(insector)) return;

      Real_v safephi;
      PhiPlaneSafety<Real_v, UnplacedStruct_t, tubeTypeT, false>(tube, point, safephi);
      vecCore::MaskedAssign(safety, !insector && safephi < kInfLength && safephi > safety, safephi);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOutOld(UnplacedStruct_t const &tube,
                                                                          Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace ::vecgeom::TubeTypes;
    using namespace TubeUtilities;

    safety          = tube.fZ - Abs(point.z());
    Real_v r        = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v safermax = tube.fRmax - r;
    vecCore::MaskedAssign(safety, safermax < safety, safermax);

    if (checkRminTreatment<tubeTypeT>(tube)) {
      Real_v safermin = r - tube.fRmin;
      vecCore::MaskedAssign(safety, safermin < safety, safermin);
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      // Now using Wedge to calculate the SafetyToOut for a sector of a tube
      Real_v safephi = tube.fPhiWedge.SafetyToOut<Real_v>(point);
      vecCore::MaskedAssign(safety, safephi < safety, safephi);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> ApproxSurfaceNormalKernel(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
  {

    Vector3D<Real_v> norm(0., 0., 0.);
    Real_v radius   = point.Perp();
    Real_v distRMax = vecCore::math::Abs(radius - unplaced.fRmax);
    Real_v distRMin = kInfLength;
    vecCore__MaskedAssignFunc(distRMax, distRMax < Real_v(0.), InfinityLength<Real_v>());
    if (unplaced.fRmin) {
      distRMin = Abs(unplaced.fRmin - radius);
      vecCore__MaskedAssignFunc(distRMin, distRMin < Real_v(0.), InfinityLength<Real_v>());
    }
    Real_v distMin = Min(distRMin, distRMax);

    Real_v distPhi1 = kInfLength, distPhi2 = kInfLength;
    if (unplaced.fDphi != vecgeom::kTwoPi) {
      distPhi1 = point.x() * unplaced.fPhiWedge.GetNormal1().x() + point.y() * unplaced.fPhiWedge.GetNormal1().y();
      distPhi2 = point.x() * unplaced.fPhiWedge.GetNormal2().x() + point.y() * unplaced.fPhiWedge.GetNormal2().y();

      vecCore__MaskedAssignFunc(distPhi1, distPhi1 < Real_v(0.), InfinityLength<Real_v>());
      vecCore__MaskedAssignFunc(distPhi2, distPhi2 < Real_v(0.), InfinityLength<Real_v>());
      distMin = Min(distMin, Min(distPhi1, distPhi2));
    }

    Real_v distZ = kInfLength;
    vecCore__MaskedAssignFunc(distZ, point.z() < Real_v(0.), vecCore::math::Abs(point.z() + unplaced.fZ));
    vecCore__MaskedAssignFunc(distZ, point.z() >= Real_v(0.), vecCore::math::Abs(point.z() - unplaced.fZ));
    distMin = Min(distMin, distZ);

    if (unplaced.fDphi) {
      Vector3D<Real_v> normal1 = unplaced.fPhiWedge.GetNormal1();
      Vector3D<Real_v> normal2 = unplaced.fPhiWedge.GetNormal2();
      vecCore__MaskedAssignFunc(norm, distMin == distPhi1, -normal1);
      vecCore__MaskedAssignFunc(norm, distMin == distPhi2, -normal2);
    }

    vecCore__MaskedAssignFunc(norm, (distMin == distZ) && (point.z() < Real_v(0.)), Vector3D<Real_v>(0., 0., -1.));
    vecCore__MaskedAssignFunc(norm, (distMin == distZ) && (point.z() >= Real_v(0.)), Vector3D<Real_v>(0., 0., 1.));

    if (vecCore::math::Abs(point.z()) < (unplaced.fZ + kTolerance)) {
      Vector3D<Real_v> temp = point;
      temp.z()              = Real_v(0.);
      vecCore__MaskedAssignFunc(norm, distMin == distRMax, temp.Unit());
      if (unplaced.fRmin) vecCore__MaskedAssignFunc(norm, distMin == distRMin, -temp.Unit());
    }

    return norm;
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &norm, Bool_v &valid)
  {

    valid = Bool_v(false);
    Bool_v isPointInside(false), isPointOutside(false);
    GenericKernelForContainsAndInside<Real_v, true>(unplaced, point, isPointInside, isPointOutside);
    if (isPointInside || isPointOutside) {
      norm = ApproxSurfaceNormalKernel<Real_v>(unplaced, point);
      return;
    }

    int nosurface = 0; // idea from trapezoid;; change nomenclature as confusing

    Precision x2y2 = Sqrt(point.x() * point.x() + point.y() * point.y());
    bool inZ = ((point.z() < unplaced.fZ + kTolerance) && (point.z() > -unplaced.fZ - kTolerance)); // in right z range
    bool inR = ((x2y2 >= unplaced.fRmin - kTolerance) && (x2y2 <= unplaced.fRmax + kTolerance));    // in right r range
    // bool inPhi = fWedge.Contains(point);
    // can we combine these two into one??
    if (inR && (Abs(point.z() - unplaced.fZ) <= kTolerance)) { // top lid, normal along +Z
      norm.Set(0., 0., 1.);
      nosurface++;
    }
    if (inR && (Abs(point.z() + unplaced.fZ) <= kTolerance)) { // bottom base, normal along -Z
      if (nosurface > 0) {
        // norm exists already; just add to it
        norm[2] += Real_v(-1.);
      } else {
        norm.Set(0., 0., -1.);
      }
      nosurface++;
    }
    if (unplaced.fRmin > 0.) {
      if (inZ && (Abs(x2y2 - unplaced.fRmin) <= kTolerance)) { // inner tube wall, normal  towards center
        Precision invx2y2 = 1. / x2y2;
        if (nosurface == 0) {
          norm[0] = -point[0] * invx2y2;
          norm[1] = -point[1] * invx2y2; // -ve due to inwards
          norm[2] = Real_v(0.);
        } else {
          norm[0] += -point[0] * invx2y2;
          norm[1] += -point[1] * invx2y2;
        }
        nosurface++;
      }
    }
    if (inZ && (Abs(x2y2 - unplaced.fRmax) <= kTolerance)) { // outer tube wall, normal outwards
      Precision invx2y2 = 1. / x2y2;
      if (nosurface > 0) {
        norm[0] += point[0] * invx2y2;
        norm[1] += point[1] * invx2y2;
      } else {
        norm[0] = point[0] * invx2y2;
        norm[1] = point[1] * invx2y2;
        norm[2] = Real_v(0.);
      }
      nosurface++;
    }

    // otherwise we get a normal from the wedge
    if (unplaced.fDphi < vecgeom::kTwoPi) {
      if (inR && unplaced.fPhiWedge.IsOnSurface1(point)) {
        if (nosurface == 0)
          norm = -unplaced.fPhiWedge.GetNormal1();
        else
          norm += -unplaced.fPhiWedge.GetNormal1();
        nosurface++;
      }
      if (inR && unplaced.fPhiWedge.IsOnSurface2(point)) {
        if (nosurface == 0)
          norm = -unplaced.fPhiWedge.GetNormal2();
        else
          norm += -unplaced.fPhiWedge.GetNormal2();
        nosurface++;
      }
    }
    if (nosurface > 1) norm = norm / std::sqrt(1. * nosurface);
    valid = nosurface != 0; // this is for testing only
  }

}; // End of struct TubeImplementation

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
