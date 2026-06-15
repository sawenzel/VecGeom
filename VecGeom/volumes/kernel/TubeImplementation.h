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

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, TubeImplementation, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace TubeUtilities {

/**
 * @brief Check whether a point lies inside a cylindrical phi sector.
 * @details The sector is defined by the two along-vectors that bound the phi
 * interval. The same test could be implemented with `atan2`, but this helper
 * avoids trigonometric work and uses only multiplications and comparisons.
 *
 * The expression `(-x * starty + y * startx) >= 0` checks whether going from
 * the start vector to the point follows the counter-clockwise direction when
 * taking the shortest turn. The expression `(-endx * y + endy * x) >= 0`
 * checks whether going from the point to the end vector also follows the
 * counter-clockwise direction.
 *
 * For sectors smaller than pi, both checks must hold: if the path
 * `start -> point -> end` is counter-clockwise, the point is inside the
 * sector. For sectors larger than pi, only one of the checks must hold,
 * because one of the two shortest turns can legitimately be clockwise even
 * while the point is still inside the larger sector.
 *
 * The helper can resolve the smaller-than-pi versus larger-than-pi choice
 * either at compile time or at runtime, depending on `ShapeType`.
 * @param volume Tube-like volume providing the phi-sector definition.
 * @param x X coordinate of the point to classify.
 * @param y Y coordinate of the point to classify.
 * @param[out] ret Classification result written by the helper.
 * @return None. The classification is written to @p ret.
 */
template <typename Real_v, typename ShapeType, typename UnplacedVolumeType, bool onSurfaceT, bool includeSurface = true>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PointInCyclicalSector(UnplacedVolumeType const &volume,
                                                                        Real_v const &x, Real_v const &y, bool &ret)
{
  using namespace ::vecgeom::TubeTypes;
  // VECGEOM_VALIDATE(SectorType<ShapeType>::value != kNoAngle, << "ShapeType without a
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
        ret = (startCheck >= -kHalfTolerance) && (endCheck >= -kHalfTolerance);
      else
        ret = (startCheck >= kHalfTolerance) && (endCheck >= kHalfTolerance);
    } else {
      if (includeSurface)
        ret = (startCheck >= -kHalfTolerance) || (endCheck >= -kHalfTolerance);
      else
        ret = (startCheck >= kHalfTolerance) || (endCheck >= kHalfTolerance);
    }
  }
}

/**
 * @brief Solve the radial trajectory intersection with a tube circle.
 * @details The caller passes the reduced quadratic coefficients for the
 * cylindrical intersection in the transverse plane. Depending on
 * @p LargestSolution, the helper keeps either the near or far root and then
 * checks the usual acceptance conditions: a sufficiently non-negative
 * distance, optional z acceptance, and optional phi-sector acceptance of the
 * hit point.
 * @param b Reduced linear coefficient of the quadratic.
 * @param c Reduced constant coefficient of the quadratic.
 * @param tube Tube geometry providing z and optional phi limits.
 * @param pos Starting point of the trajectory.
 * @param dir Direction of the trajectory.
 * @param[out] dist Accepted intersection distance.
 * @param[out] ok Validity flag for the computed root.
 * @return None. The result is written to @p dist and @p ok.
 */
template <typename Real_v, typename UnplacedStruct_t, typename TubeType, bool LargestSolution, bool insectorCheck>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void CircleTrajectoryIntersection(Real_v const &b, Real_v const &c,
                                                                               UnplacedStruct_t const &tube,
                                                                               Vector3D<Real_v> const &pos,
                                                                               Vector3D<Real_v> const &dir,
                                                                               Real_v &dist, bool &ok)
{
  using namespace ::vecgeom::TubeTypes;

  Real_v delta = b * b - c;
  ok           = delta > Real_v(0.);
  if (LargestSolution) ok |= delta == Real_v(0.); // this takes care of scratching conventions

  if (!ok) {
    dist = Real_v(0.);
    return;
  }
  delta = Sqrt(delta);
  if (!LargestSolution) delta = -delta;

  dist = -b + delta;
  // A.G There may be points propagated to Rmax+tolerance which get here and NEED to se a valid negative crossing
  // at distance > tolerance, so we need to enlarge the tolerance
  ok &= dist >= -2 * kTolerance;
  if (!ok) return;

  if (insectorCheck) {
    Real_v hitz = pos.z() + dist * dir.z();
    ok &= (Abs(hitz) <= tube.fZ);
    if (!ok) return;

    if (checkPhiTreatment<TubeType>(tube)) {
      bool insector = false;
      Real_v hitx   = pos.x() + dist * dir.x();
      Real_v hity   = pos.y() + dist * dir.y();
      PointInCyclicalSector<Real_v, TubeType, UnplacedStruct_t, false, true>(tube, hitx, hity, insector);
      // insector = tube.fPhiWedge.ContainsWithBoundary<Real_v>(
      // Vector3D<Real_v>(hitx, hity, hitz) );
      ok &= insector;
    }
  }
}

/**
 * @brief Return the perpendicular distance from a 2D point to a 2D unit
 * direction.
 * @details Let `p` be the point vector and `v` the unit direction of the
 * infinite line. If `theta` is the angle between them, then the perpendicular
 * distance is `|p| * sin(theta)`. The 2D cross-product magnitude is
 * `|p x v| = |p| * |v| * sin(theta)`, and because `|v| = 1` this is exactly
 * the perpendicular distance. In coordinates, this reduces to
 * `p.x * v.y - p.y * v.x`.
 * @param px X coordinate of the point.
 * @param py Y coordinate of the point.
 * @param vx X component of the unit direction.
 * @param vy Y component of the unit direction.
 * @return The signed 2D cross-product magnitude, equal to the perpendicular
 * distance up to orientation.
 */
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v PerpDist2D(Real_v const &px, Real_v const &py, Real_v const &vx,
                                                               Real_v const &vy)
{
  return px * vy - py * vx;
}

/**
 * @brief Compute the safety from a point to the active phi boundary plane.
 * @details For sectors larger than pi, the initial safety is the transverse
 * radius because the nearest limiting phi plane can be farther than either
 * direct signed plane distance. For smaller sectors, the helper starts from
 * infinity and tightens the result with the valid phi-plane distances. The
 * sign convention is adjusted depending on whether the point is already known
 * to be inside or outside the phi sector.
 * @param tube Tube geometry providing the phi-plane definition.
 * @param pos Point for which the phi safety is evaluated.
 * @param[out] safety Safety value updated by the helper.
 * @return None. The result is written to @p safety.
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
    if (absphi1 > kHalfTolerance) safety = absphi1;
    return;
  }

  // make sure point falls on positive part of projection
  if (phi1 > -kHalfTolerance && phi1 < safety) safety = phi1;

  Real_v phi2 = PerpDist2D<Real_v>(pos.x(), pos.y(), Real_v(tube.fAlongPhi2x), Real_v(tube.fAlongPhi2y));
  if (!inside) phi2 *= -1;

  // make sure point falls on positive part of projection
  if (phi2 > -kHalfTolerance && phi2 < safety) safety = phi2;
}

/**
 * @brief Intersect a trajectory with a phi boundary plane.
 * @details All points on the phi-plane along-vector lie on
 * `s * (alongX, alongY)`, while the particle trajectory lies on
 * `(x, y) + t * (vx, vy)`. Solving
 * `s * (alongX, alongY) == (x, y) + t * (vx, vy)` for `t` gives
 * `t = (alongY * x - alongX * y) / (vy * alongX - vx * alongY)`.
 *
 * When requested, the helper also checks that the intersection stays within
 * the tube z/r limits and that it lies on the positive direction of the phi
 * vector, i.e. `hitx * alongX + hity * alongY > 0`.
 * @param alongX X component of the phi-boundary along-vector.
 * @param alongY Y component of the phi-boundary along-vector.
 * @param normX X component of the inward phi-plane normal.
 * @param normY Y component of the inward phi-plane normal.
 * @param tube Tube geometry providing z/r bounds.
 * @param pos Starting point of the trajectory.
 * @param dir Direction of the trajectory.
 * @param[out] dist Intersection distance along the trajectory.
 * @param[out] ok Validity flag for the computed intersection.
 * @return None. The result is written to @p dist and @p ok.
 */
template <typename Real_v, typename UnplacedStruct_t, typename TubeType, bool PositiveDirectionOfPhiVector,
          bool insectorCheck>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void PhiPlaneTrajectoryIntersection(
    Precision alongX, Precision alongY, Precision normX, Precision normY, UnplacedStruct_t const &tube,
    Vector3D<Real_v> const &pos, Vector3D<Real_v> const &dir, Real_v &dist, bool &ok)
{

  dist = kInfLength;

  // approaching phi plane from the right side?
  // this depends whether we use it for DistanceToIn or DistanceToOut
  // Note: wedge normals point towards the wedge inside, by convention!
  Real_v dirDotNorm = dir.x() * normX + dir.y() * normY;
  if (insectorCheck)
    ok = (dirDotNorm > Real_v(0.)); // DistToIn  -- require tracks entering volume
  else
    ok = (dirDotNorm < Real_v(0.)); // DistToOut -- require tracks leaving volume

  Real_v dirDotXY = (dir.y() * alongX - dir.x() * alongY);
  dist            = (alongY * pos.x() - alongX * pos.y()) / NonZero(dirDotXY);
  // A.G to check validity, we have to compare with tolerance the safety rather than the distance to plane
  ok &= (dist * Abs(dirDotNorm)) > -kHalfTolerance;
  if (ok && dist < Real_v(0.)) dist = Real_v(0.);

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

/**
 * @brief Check whether a point lies on the selected cylindrical surface.
 * @details The helper accepts points within the corresponding radial
 * tolerance band and inside the allowed z range. `ForInnerSurface` selects the
 * inner cylindrical wall of a hollow tube or the outer cylindrical wall.
 * @param tube Tube geometry providing the radial and longitudinal limits.
 * @param point Point to test.
 * @return `true` when the point lies on the requested cylindrical surface.
 */
template <typename Real_v, typename UnplacedStruct_t, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsOnTubeSurface(UnplacedStruct_t const &tube,
                                                                  Vector3D<Real_v> const &point)
{
  const Real_v rho = point.Perp2();
  if (ForInnerSurface) {
    return (rho >= tube.fTolOrmin2) && (rho <= tube.fTolIrmin2) && (Abs(point.z()) < (tube.fZ + kTolerance));
  } else {
    return (rho >= tube.fTolIrmax2) && (rho <= tube.fTolOrmax2) && (Abs(point.z()) < (tube.fZ + kTolerance));
  }
}

/**
 * @brief Return the inward or outward radial normal of a tube surface point.
 * @details The helper constructs the unnormalized cylindrical normal in the
 * transverse plane. `ForInnerSurface` flips the sign so that the normal points
 * toward the tube interior for the inner wall and away from it for the outer
 * wall.
 * @param point Surface point used to build the radial direction.
 * @return Unnormalized radial normal for the requested cylindrical surface.
 */
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

/**
 * @brief Check whether a surface point moves into the requested tube wall.
 * @details The helper combines the surface predicate with the sign of the
 * trajectory projected on the corresponding cylindrical normal.
 * `ForInnerSurface` selects the inner or outer cylindrical wall.
 * @param tube Tube geometry providing the tolerance bands.
 * @param point Surface point to test.
 * @param direction Track direction at the surface point.
 * @return `true` when the point is on the selected surface and the track moves
 * into the tube through that wall.
 */
template <typename Real_v, typename UnplacedStruct_t, bool ForInnerSurface>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsMovingInsideTubeSurface(UnplacedStruct_t const &tube,
                                                                            Vector3D<Real_v> const &point,
                                                                            Vector3D<Real_v> const &direction)
{
  return IsOnTubeSurface<Real_v, UnplacedStruct_t, ForInnerSurface>(tube, point) &&
         (direction.Dot(GetNormal<Real_v, ForInnerSurface>(point)) < -0.5 * int(!ForInnerSurface) * kTolerance);
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

  /**
   * @brief Shared scalar kernel for `Contains` and `Inside`.
   * @details The kernel classifies the point against z, outer radius, inner
   * radius, and optional phi limits, and reports whether the point is fully
   * inside or fully outside according to the tolerance conventions expected by
   * the public wrappers.
   * @param tube Tube geometry to classify against.
   * @param point Point to classify.
   * @param[out] completelyinside Set when the point is strictly inside all
   * active boundaries.
   * @param[out] completelyoutside Set when the point is strictly outside any
   * active boundary.
   * @return None. The classification is written to the output flags.
   */
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &tube, Vector3D<Real_v> const &point, bool &completelyinside, bool &completelyoutside)
  {
    using namespace ::vecgeom::TubeTypes;

    // very fast check on z-height
    Real_v absz       = Abs(point[2]);
    completelyoutside = absz > MakePlusTolerant<true>(tube.fZ);
    if (ForInside) completelyinside = absz < MakeMinusTolerant<true>(tube.fZ);
    if (completelyoutside) return;

    // check on RMAX
    Real_v r2 = point.x() * point.x() + point.y() * point.y();
    completelyoutside |= r2 > MakePlusTolerantSquare<true>(tube.fRmax);
    if (ForInside) completelyinside &= r2 < MakeMinusTolerantSquare<true>(tube.fRmax);
    if (completelyoutside) return;

    // check on RMIN
    if (checkRminTreatment<tubeTypeT>(tube)) {
      completelyoutside |= r2 <= MakeMinusTolerantSquare<true>(tube.fRmin);
      if (ForInside) completelyinside &= r2 > MakePlusTolerantSquare<true>(tube.fRmin);
      if (completelyoutside) return;
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      bool completelyoutsidephi = false;
      bool completelyinsidephi  = false;
      TubeUtilities::PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false, false>(
          tube, point.x(), point.y(), completelyinsidephi);
      TubeUtilities::PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false, true>(tube, point.x(), point.y(),
                                                                                             completelyoutsidephi);
      completelyoutsidephi = !completelyoutsidephi;

      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  /**
   * @brief Test whether a point is contained in the tube.
   * @details This is the boolean wrapper around
   * `GenericKernelForContainsAndInside`.
   * @param tube Tube geometry to test against.
   * @param point Point to classify.
   * @param[out] contains Set to `true` when the point is not outside.
   * @return None. The result is written to @p contains.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &tube,
                                                                    Vector3D<Real_v> const &point, bool &contains)
  {
    bool unused  = false;
    bool outside = false;
    GenericKernelForContainsAndInside<Real_v, false>(tube, point, unused, outside);
    contains = !outside;
  }

  /**
   * @brief Classify a point as inside, outside, or on the surface.
   * @details This wrapper translates the boolean classification produced by
   * `GenericKernelForContainsAndInside` into the `EInside` convention.
   * @param tube Tube geometry to test against.
   * @param point Point to classify.
   * @param[out] inside Classification result.
   * @return None. The result is written to @p inside.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &tube,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside  = false;
    bool completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, true>(tube, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = EInside::kOutside;
    if (completelyinside) inside = EInside::kInside;
  }

  /**
   * @brief Compute the entry distance from a point to the tube.
   * @details If the point is farther than one hundred times the maximum tube
   * extent from the origin, the method first moves it closer along the track
   * and then delegates to `DistanceToInKernel`. The final answer is the manual
   * pre-step plus the kernel result. This keeps the quadratic root evaluation
   * stable without reintroducing the old Newton fallback in
   * `CircleTrajectoryIntersection`, and matches the `ShapeTester` behavior for
   * very distant points.
   * @param tube Tube geometry to intersect.
   * @param pointt Starting point of the trajectory.
   * @param dir Direction of the trajectory.
   * @param stepMax Maximum step requested by the caller.
   * @param[out] distance Computed distance to the first valid entry.
   * @return None. The result is written to @p distance.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &tube,
                                                                        Vector3D<Real_v> const &pointt,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    Vector3D<Real_v> point = pointt;
    const Real_v ptDist    = point.Mag();
    Real_v distToMove(0.);
    const Precision order = 100.;
    if (ptDist > order * tube.fMaxVal) {
      distToMove = ptDist - Real_v(order * tube.fMaxVal);
      point += distToMove * dir;
    }
    DistanceToInKernel<Real_v>(tube, point, dir, stepMax, distance);
    distance += distToMove;
  }

  /**
   * @brief Compute the entry distance once far-away points are already handled.
   * @details The kernel rejects trajectories that are clearly moving away,
   * checks whether the point is already inside, then considers valid
   * intersections with z planes, cylindrical surfaces, and phi planes and
   * keeps the closest accepted entry.
   * @param tube Tube geometry to intersect.
   * @param point Starting point of the trajectory.
   * @param dir Direction of the trajectory.
   * @param stepMax Maximum step requested by the caller.
   * @param[out] distance Computed distance to the first valid entry, `-1` when
   * the point is already inside, or `kInfLength` when no entry exists.
   * @return None. The result is written to @p distance.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToInKernel(UnplacedStruct_t const &tube,
                                                                              Vector3D<Real_v> const &point,
                                                                              Vector3D<Real_v> const &dir,
                                                                              Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using namespace TubeUtilities;
    using namespace ::vecgeom::TubeTypes;
    bool done = false;

    //=== First, for points outside and moving away --> return infinity
    distance = kInfLength;

    const bool hasZPlanes = tube.fZ < kInfLength;

    // outside of Z range and going away?
    Real_v distz(0.);
    if (hasZPlanes) {
      distz = Abs(point.z()) - tube.fZ; // avoid a division for now
      done  = distz > kHalfTolerance && point.z() * dir.z() >= 0;
      if (done) return;
    }

    // outside of outer tube and going away?
    Real_v rsq   = point.x() * point.x() + point.y() * point.y();
    Real_v rdotn = point.x() * dir.x() + point.y() * dir.y();
    Real_v nsq   = Real_v(1.) - dir.z() * dir.z();
    done         = rsq > tube.fTolIrmax2 && rdotn >= 0;
    if (done) return;

    //=== Next, check all dimensions of the tube, whether points are inside -->
    // return -1
    distance = Real_v(-1.0);

    // Infinite-z helper uses such as CutTube never reject entry on z here.
    bool inside = !hasZPlanes || distz < -kHalfTolerance;

    inside &= rsq < tube.fTolIrmax2;
    if (checkRminTreatment<tubeTypeT>(tube)) {
      inside &= rsq > tube.fTolIrmin2;
    }
    if (checkPhiTreatment<tubeTypeT>(tube) && inside) {
      bool insector = false;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false, false>(tube, point.x(), point.y(), insector);
      inside &= insector;
      // inside &= tube.fPhiWedge.ContainsWithoutBoundary<Real_v>( point );  //
      // slower than PointInCyclicalSector()
    }
    done = inside;
    if (done) return;

    //=== Next step: check if z-plane is the right entry point (both r,phi
    // should be valid at z-plane crossing)
    distance = Real_v(kInfLength);

    if (hasZPlanes) {
      const bool enteringZPlane = point.z() * dir.z() < Real_v(0.);
      const bool zSurfaceEntry  = distz <= Real_v(0.) && Abs(distz) <= kHalfTolerance && enteringZPlane &&
                                  Abs(dir.z()) * tube.fZ > kToleranceDist<Real_v>;
      distz                     = zSurfaceEntry ? Real_v(0.) : distz / NonZeroAbs(dir.z());
      // std::cerr << "Dist : " << distz << std::endl;

      Real_v hitx = point.x() + distz * dir.x();
      Real_v hity = point.y() + distz * dir.y();
      Real_v r2   = hitx * hitx + hity * hity; // radius of intersection with z-plane
      // Surface cap entries are decided from plane distance before dividing
      // by the shallow z projection, otherwise tolerated starts can look like
      // sizeable negative path lengths.
      bool okz = (zSurfaceEntry || distz > -kHalfTolerance) && enteringZPlane;

      okz &= (r2 <= tube.fRmax2);
      if (checkRminTreatment<tubeTypeT>(tube)) {
        okz &= (tube.fRmin2 <= r2);
      }
      if (checkPhiTreatment<tubeTypeT>(tube) && okz) {
        bool insector = false;
        PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false>(tube, hitx, hity, insector);
        okz &= insector;
        // okz &= tube.fPhiWedge.ContainsWithBoundary<Real_v>(
        // Vector3D<Real_v>(hitx, hity, 0.0) );
      }
      if (okz) {
        distance = distz;
        done     = true;
      }
    }

    const Real_v absz = Abs(point.z());

    // point on outer cyl?
    const bool isOnOuterSurface = rsq >= tube.fTolIrmax2 && rsq <= tube.fTolOrmax2 && absz < (tube.fZ + kTolerance);
    bool movingInsideOuter      = false;
    if (isOnOuterSurface) {
      const Real_v radialProjectionTolerance = Real_v(0.5) * kToleranceDist<Real_v> * nsq;
      movingInsideOuter                      = rdotn < -radialProjectionTolerance;
    }
    if (isOnOuterSurface && !movingInsideOuter) {
      distance = kInfLength;
      return;
    }
    if (done) return;

    bool isOnSurfaceAndMovingInside = isOnOuterSurface && movingInsideOuter;
    if (checkRminTreatment<tubeTypeT>(tube)) {
      // point on inner cyl?
      const bool isOnInnerSurface = rsq >= tube.fTolOrmin2 && rsq <= tube.fTolIrmin2 && absz < (tube.fZ + kTolerance);
      const Real_v radialProjectionTolerance = Real_v(0.5) * kToleranceDist<Real_v> * nsq;
      isOnSurfaceAndMovingInside |= isOnInnerSurface && rdotn >= -radialProjectionTolerance;
    }

    if (!checkPhiTreatment<tubeTypeT>(tube)) {
      if (isOnSurfaceAndMovingInside) {
        distance = Real_v(0.);
        return;
      }
    } else {
      bool insector = false;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false>(tube, point.x(), point.y(), insector);
      if (insector && isOnSurfaceAndMovingInside) {
        distance = Real_v(0.);
        return;
      }
    }

    //=== Next step: intersection of the trajectories with the two circles

    // Here for values used in both rmin and rmax calculations
    Real_v invnsq = Real_v(1.) / NonZero(nsq);
    Real_v b      = invnsq * rdotn;

    /*
     * rmax
     * If the particle were to hit rmax, it would hit the closest point of the
     * two
     * --> only consider the smallest solution of the quadratic equation
     */
    Real_v crmax     = invnsq * (rsq - tube.fRmax2);
    Real_v dist_rmax = kInfLength;
    bool ok_rmax     = false;
    CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, true>(b, crmax, tube, point, dir,
                                                                                   dist_rmax, ok_rmax);
    if (ok_rmax && dist_rmax < distance) {
      distance = dist_rmax;
      return;
    }

    /*
     * rmin
     * If the particle were to hit rmin, it would hit the farthest point of the
     * two
     * --> only consider the largest solution to the quadratic equation
     */
    Real_v dist_rmin = -kInfLength;
    bool ok_rmin     = false;
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
      if (ok_rmin && dist_rmin < distance) distance = dist_rmin;
      // done |= ok_rmin; // can't be done here, it's wrong in case
      // phi-treatment is needed!
    }

    /*
     * Calculate intersection between trajectory and the two phi planes
     */
    if (checkPhiTreatment<tubeTypeT>(tube)) {

      Real_v dist_phi;
      bool ok_phi   = false;
      auto const &w = tube.fPhiWedge;
      PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, SectorType<tubeTypeT>::value != kOnePi, true>(
          tube.fAlongPhi1x, tube.fAlongPhi1y, w.GetNormal1().x(), w.GetNormal1().y(), tube, point, dir, dist_phi,
          ok_phi);
      if (ok_phi && dist_phi < distance) distance = dist_phi;

      /*
       * If the tube is pi degrees, there's just one phi plane,
       * so no need to check again
       */

      if (SectorType<tubeTypeT>::value != kOnePi) {
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, true>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;
      }
    }
  } // end of DistanceToIn()

  /**
   * @brief Compute the exit distance from a point inside the tube.
   * @details The method rejects points already outside, then evaluates the
   * candidate exits through z planes, cylindrical surfaces, and optional phi
   * planes and keeps the smallest valid distance according to the tube surface
   * conventions.
   * @param tube Tube geometry to intersect.
   * @param point Starting point of the trajectory.
   * @param dir Direction of the trajectory.
   * @param stepMax Maximum step requested by the caller.
   * @param[out] distance Computed distance to the first valid exit, or `-1`
   * when the point is not inside the tube.
   * @return None. The result is written to @p distance.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &tube,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using namespace ::vecgeom::TubeTypes;
    using namespace TubeUtilities;

    distance = Real_v(-1.);

    //=== First we check all dimensions of the tube, whether points are outside
    //--> return -1

    // For points outside z-range, return -1
    Real_v distz = tube.fZ - Abs(point.z()); // avoid a division for now
    if (distz < -kHalfTolerance) return;     // distance is already set to -1

    Real_v rsq   = point.x() * point.x() + point.y() * point.y();
    Real_v rdotn = dir.x() * point.x() + dir.y() * point.y();
    Real_v crmax = rsq - tube.fRmax2; // avoid a division for now
    Real_v crmin = rsq;

    // if outside of Rmax, return -1
    if (crmax > Real_v(2.0 * kTolerance * tube.fRmax)) return;

    if (checkRminTreatment<tubeTypeT>(tube)) {
      // if point is within inner-hole of a hollow tube, it is outside of the
      // tube --> return -1
      crmin -= tube.fRmin2; // avoid a division for now
      if (crmin < Real_v(-2.0 * kTolerance * tube.fRmin)) return;
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      bool insector = false;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false>(tube, point.x(), point.y(), insector);
      if (!insector) return;
    }

    // OK, since we're here, then distance must be non-negative, and the
    // smallest of possible intersections
    distance = Real_v(kInfLength);

    Real_v invdirz = Real_v(1.) / NonZero(dir.z());
    distz          = (dir.z() < 0) ? (-tube.fZ - point.z()) * invdirz : (tube.fZ - point.z()) * invdirz;
    if (Abs(invdirz) < InvdirNearParallel(tube.fRmax) && distz < distance) distance = distz;

    /*
     * Find the intersection of the trajectories with the two circles.
     * Here I compute values used in both rmin and rmax calculations.
     */

    Real_v invnsq = Real_v(1.) / NonZero(Real_v(1.) - dir.z() * dir.z());
    Real_v b      = invnsq * rdotn;
    // Ignore cylindrical surface crossings for directions near-parallel to Z
    // The upper limit matches the direction for which a point on the surface could still hit the cylinder before
    // hitting the Z plane
    bool checkTube = invnsq < tube.fZ * tube.fZ * kInvTolerance * kInvTolerance;

    /*
     * rmin
     */

    if (checkTube && checkRminTreatment<tubeTypeT>(tube)) {
      bool isOnInnerSurface = false;
      if (crmin <= tube.fTolIrmin2 - tube.fRmin2) {
        isOnInnerSurface = crmin >= tube.fTolOrmin2 - tube.fRmin2;
      }
      if (isOnInnerSurface) {
        const Real_v nsq                       = Real_v(1.) - dir.z() * dir.z();
        const Real_v radialProjectionTolerance = Real_v(0.5) * kToleranceDist<Real_v> * nsq;
        if (rdotn < -radialProjectionTolerance) distance = Real_v(0.);
      } else {
        Real_v dist_rmin = kInfLength;
        bool ok_rmin     = false;
        crmin *= invnsq;
        CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(b, crmin, tube, point, dir,
                                                                                        dist_rmin, ok_rmin);
        if (ok_rmin && dist_rmin < distance) distance = dist_rmin;
      }
    }

    /*
     * rmax
     */

    if (checkTube) {
      bool isOnOuterSurface = false;
      if (crmax >= tube.fTolIrmax2 - tube.fRmax2) {
        isOnOuterSurface = crmax <= tube.fTolOrmax2 - tube.fRmax2;
      }
      bool exitsOuterSurfaceImmediately = false;
      if (isOnOuterSurface) {
        const Real_v nsq             = Real_v(1.) - dir.z() * dir.z();
        exitsOuterSurfaceImmediately = rdotn >= -Real_v(0.5) * kToleranceDist<Real_v> * nsq;
      }
      if (exitsOuterSurfaceImmediately) {
        distance = Real_v(0.);
      } else {
        Real_v dist_rmax = kInfLength;
        bool ok_rmax     = false;
        crmax *= invnsq;
        CircleTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, false>(b, crmax, tube, point, dir,
                                                                                       dist_rmax, ok_rmax);
        if (ok_rmax && dist_rmax < distance) distance = dist_rmax;
      }
    }

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
      bool ok_phi = false;

      auto const &w = tube.fPhiWedge;
      if (SectorType<tubeTypeT>::value == kSmallerThanPi) {

        Precision normal1X = w.GetNormal1().x();
        Precision normal1Y = w.GetNormal1().y();
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(
            tube.fAlongPhi1x, tube.fAlongPhi1y, normal1X, normal1Y, tube, point, dir, dist_phi, ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;

        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;
      } else if (SectorType<tubeTypeT>::value == kOnePi) {
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, false, false>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;
      } else {
        // angle bigger than pi or unknown
        // need to check that point falls on positive direction of phi-vectors
        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, false>(
            tube.fAlongPhi1x, tube.fAlongPhi1y, w.GetNormal1().x(), w.GetNormal1().y(), tube, point, dir, dist_phi,
            ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;

        PhiPlaneTrajectoryIntersection<Real_v, UnplacedStruct_t, tubeTypeT, true, false>(
            tube.fAlongPhi2x, tube.fAlongPhi2y, w.GetNormal2().x(), w.GetNormal2().y(), tube, point, dir, dist_phi,
            ok_phi);
        if (ok_phi && dist_phi < distance) distance = dist_phi;
      }
    }
    return;
  }

  /**
   * @brief Compute the safety distance from an external point to the tube.
   * @details The safety is the maximum of the active z, radial, and optional
   * phi distances needed to reach the tube.
   * @param tube Tube geometry to test against.
   * @param point External point.
   * @param[out] safety Computed safety-to-in value.
   * @return None. The result is written to @p safety.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &tube,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace ::vecgeom::TubeTypes;
    using namespace TubeUtilities;

    safety = Abs(point.z()) - tube.fZ;

    Real_v r        = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v safermax = r - tube.fRmax;
    if (safermax > safety) safety = safermax;

    if (checkRminTreatment<tubeTypeT>(tube)) {
      Real_v safermin = tube.fRmin - r;
      if (safermin > safety) safety = safermin;
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      bool insector = false;
      PointInCyclicalSector<Real_v, tubeTypeT, UnplacedStruct_t, false, false>(tube, point.x(), point.y(), insector);
      if (insector) return;

      Real_v safephi;
      PhiPlaneSafety<Real_v, UnplacedStruct_t, tubeTypeT, false>(tube, point, safephi);
      if (safephi < kInfLength && safephi > safety) safety = safephi;
    }
  }

  /**
   * @brief Compute the safety distance from an internal point to the tube
   * boundary.
   * @details The safety is the minimum of the active z, radial, and optional
   * phi distances needed to leave the tube.
   * @param tube Tube geometry to test against.
   * @param point Internal point.
   * @param[out] safety Computed safety-to-out value.
   * @return None. The result is written to @p safety.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &tube,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    using namespace ::vecgeom::TubeTypes;

    safety          = tube.fZ - Abs(point.z());
    Real_v r        = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v safermax = tube.fRmax - r;
    if (safermax < safety) safety = safermax;

    if (checkRminTreatment<tubeTypeT>(tube)) {
      Real_v safermin = r - tube.fRmin;
      if (safermin < safety) safety = safermin;
    }

    if (checkPhiTreatment<tubeTypeT>(tube)) {
      Real_v safephi = tube.fPhiWedge.SafetyToOut<Real_v>(point);
      if (safephi < safety) safety = safephi;
    }
  }

  /**
   * @brief Approximate a surface normal for points not classified exactly on a
   * single boundary.
   * @details The helper compares the point against the nearby z, radial, and
   * optional phi boundaries and returns the normal of the closest candidate.
   * It is mainly used as a fallback when the precise surface classification is
   * not available.
   * @param unplaced Tube geometry providing the boundary definitions.
   * @param point Point near the surface.
   * @return Approximate outward normal of the closest tube boundary.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> ApproxSurfaceNormalKernel(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
  {
    Vector3D<Real_v> norm(0., 0., 0.);
    Real_v radius   = point.Perp();
    Real_v distRMax = vecCore::math::Abs(radius - unplaced.fRmax);
    Real_v distRMin = kInfLength;
    if (unplaced.fRmin) {
      distRMin = Abs(unplaced.fRmin - radius);
    }
    Real_v distMin = Min(distRMin, distRMax);

    Real_v distPhi1 = kInfLength, distPhi2 = kInfLength;
    if (unplaced.fDphi != vecgeom::kTwoPi) {
      distPhi1 = point.x() * unplaced.fPhiWedge.GetNormal1().x() + point.y() * unplaced.fPhiWedge.GetNormal1().y();
      distPhi2 = point.x() * unplaced.fPhiWedge.GetNormal2().x() + point.y() * unplaced.fPhiWedge.GetNormal2().y();

      if (distPhi1 < Real_v(0.)) distPhi1 = kInfLength;
      if (distPhi2 < Real_v(0.)) distPhi2 = kInfLength;
      distMin = Min(distMin, Min(distPhi1, distPhi2));
    }

    Real_v distZ = point.z() < Real_v(0.) ? vecCore::math::Abs(point.z() + unplaced.fZ)
                                          : vecCore::math::Abs(point.z() - unplaced.fZ);
    distMin      = Min(distMin, distZ);

    if (unplaced.fDphi) {
      Vector3D<Real_v> normal1 = unplaced.fPhiWedge.GetNormal1();
      Vector3D<Real_v> normal2 = unplaced.fPhiWedge.GetNormal2();
      if (distMin == distPhi1) norm = -normal1;
      if (distMin == distPhi2) norm = -normal2;
    }

    if (distMin == distZ) norm = point.z() < Real_v(0.) ? Vector3D<Real_v>(0., 0., -1.) : Vector3D<Real_v>(0., 0., 1.);

    if (vecCore::math::Abs(point.z()) < (unplaced.fZ + kTolerance)) {
      Vector3D<Real_v> temp = point;
      temp.z()              = Real_v(0.);
      if (distMin == distRMax) norm = temp.Unit();
      if (unplaced.fRmin && distMin == distRMin) norm = -temp.Unit();
    }

    return norm;
  }

  /**
   * @brief Compute the exact tube normal when the point is on a known surface.
   * @details The method first checks whether the point is clearly inside or
   * outside and falls back to `ApproxSurfaceNormalKernel` in that case.
   * Otherwise it accumulates the normals of all tube boundaries touched by the
   * point and normalizes the result for edge and corner configurations.
   * @param unplaced Tube geometry providing the surface definitions.
   * @param point Point on or near the surface.
   * @param[out] norm Computed surface normal.
   * @param[out] valid Set when at least one matching surface was found.
   * @return None. The result is written to @p norm and @p valid.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &norm, bool &valid)
  {

    valid               = false;
    bool isPointInside  = false;
    bool isPointOutside = false;
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
