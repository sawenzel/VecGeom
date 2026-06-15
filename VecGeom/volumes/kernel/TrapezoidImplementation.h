//===-- kernel/TrapezoidImplementation.h ----------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file TrapezoidImplementation.h
/// @brief Navigation kernels for the trapezoid solid.
/// @author Guilherme Lima (lima@fnal.gov)
//===--------------------------------------------------------------------------===//

/// History notes:
/// 2014-05-20: Created from USolids' UTrap algorithms (G. Lima)
/// 2016-07-22: Revision and migration to the backend structure (G. Lima)

#ifndef VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/TrapezoidStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TrapezoidImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TrapezoidImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTrapezoid;
class UnplacedTrapezoid;

/// @brief Implements navigation kernels for `UnplacedTrapezoid`.
/// @details The runtime data provides the z half-length and four cached
/// outward side planes. Signed side-plane distances are negative inside,
/// positive outside, and tolerance bands are resolved by each navigation entry
/// point according to the queried convention.
struct TrapezoidImplementation {

  using PlacedShape_t    = PlacedTrapezoid;
  using UnplacedStruct_t = TrapezoidStruct<Precision>;
  using UnplacedVolume_t = UnplacedTrapezoid;
#ifndef VECGEOM_PLANESHELL
  using TrapSidePlane = TrapezoidStruct<Precision>::TrapSidePlane;
#endif

  /// @brief Evaluate side-plane signed distances and direction projections.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data containing the cached side planes.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @param[out] pdist Signed distance to each side plane; positive is outside.
  /// @param[out] proj Projection of @p dir on each outward side-plane normal.
  /// @param[out] vdist Plane-crossing distance `-pdist/proj` for each side.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void EvaluateTrack(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir, Real_v *pdist,
                                                                         Real_v *proj, Real_v *vdist)
  {
#ifdef VECGEOM_PLANESHELL
    auto const *fPlanes = unplaced.GetPlanes();
    for (unsigned int i = 0; i < 4; ++i) {
      pdist[i] = fPlanes->fA[i] * point.x() + fPlanes->fB[i] * point.y() + fPlanes->fC[i] * point.z() + fPlanes->fD[i];
      proj[i]  = fPlanes->fA[i] * dir.x() + fPlanes->fB[i] * dir.y() + fPlanes->fC[i] * dir.z();
      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }
#else
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    // loop over side planes - find pdist,proj for each side plane
    // auto-vectorizable part of loop
    for (unsigned int i = 0; i < 4; ++i) {
      // Note: normal vector is pointing outside the volume (convention), therefore
      // pdist>0 if point is outside  and  pdist<0 means inside
      pdist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;

      // proj is projection of dir over the normal vector of side plane, hence
      // proj > 0 if pointing ~same direction as normal and proj<0 if ~opposite to normal
      proj[i] = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();

      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }
#endif
  }

  /// @brief Test whether a local point is contained in or on the trapezoid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data.
  /// @param point Local point to test.
  /// @param[out] inside Set to true unless @p point is outside a tolerated limiting plane.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused = false, outside = false;
    GenericKernelForContainsAndInside<Real_v, false>(unplaced, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Inside_t Integer-like type used for `EInside` values.
  /// @param unplaced Trapezoid data.
  /// @param point Local point to classify.
  /// @param[out] inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyInside = false, completelyOutside = false;
    GenericKernelForContainsAndInside<Real_v, true>(unplaced, point, completelyInside, completelyOutside);

    inside = Inside_t(EInside::kSurface);
    if (completelyOutside) inside = Inside_t(EInside::kOutside);
    if (completelyInside) inside = Inside_t(EInside::kInside);
  }

  /// @brief Shared classification helper for containment and inside queries.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam ForInside When true, also compute the strict-inside flag.
  /// @param unplaced Trapezoid data.
  /// @param point Local point to classify.
  /// @param[out] completelyInside Set when the point is separated from all limiting planes by the inside tolerance.
  /// @param[out] completelyOutside Set when the point is outside any limiting plane beyond tolerance.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, bool &completelyInside, bool &completelyOutside)
  {
    // z-region
    completelyOutside = Abs(point[2]) > MakePlusTolerant<true>(unplaced.fDz);
    if (ForInside) {
      completelyInside = Abs(point[2]) < MakeMinusTolerant<true>(unplaced.fDz);
    }

#ifdef VECGEOM_PLANESHELL
    unplaced.GetPlanes()->GenericKernelForContainsAndInside<Real_v, true>(point, completelyInside, completelyOutside);
#else
    // here for PLANESHELL=OFF (disabled)
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    Real_v dist[4];
    for (unsigned int i = 0; i < 4; ++i) {
      dist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    }

    for (unsigned int i = 0; i < 4; ++i) {
      // is it outside of this side plane?
      completelyOutside = completelyOutside || dist[i] > Real_v(MakePlusTolerant<true>(0.));
      if (ForInside) {
        completelyInside = completelyInside && dist[i] < Real_v(MakeMinusTolerant<true>(0.));
      }
    }
#endif
  }

  /// @brief Compute the first entry distance from outside the trapezoid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Entry distance, `kInfLength` on miss, or `-1` for wrong-side candidates.
  ///
  /// @details For each limiting component (z slab and side planes), the
  /// algorithm computes the ray interval for which the track is inside that
  /// component. The entry point is the largest lower bound (`smin`), and the
  /// exit point is the smallest upper bound (`smax`). If the largest lower
  /// bound exceeds the smallest upper bound, the trajectory misses the shape.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    distance = kInfLength;

    //
    // Step 1: find range of distances along dir between Z-planes (smin, smax)
    //

    // step 1.a) input particle is moving away --> return infinity
    Real_v signZdir = Sign(dir.z());
    Real_v max      = signZdir * unplaced.fDz - point.z(); // z-dist to farthest z-plane

    // done = done || (dir.z()>0.0 && max < MakePlusTolerant<true>(0.));  // check if moving away towards +z
    // done = done || (dir.z()<0.0 && max > MakeMinusTolerant<true>(0.)); // check if moving away towards -z
    bool done = signZdir * max < Real_v(MakePlusTolerant<true>(0.0)); // if outside + moving away towards +/-z

    // if all particles moving away, we're done
    if (done) return;

    // Step 1.b) General case:
    //   smax,smin are range of distances within z-range, taking direction into account.
    //   smin<smax - smax is positive, but smin may be either positive or negative
    Real_v invdir = Real_v(1.0) / NonZero(dir.z()); // convert distances from z to dir
    Real_v smax   = max * invdir;
    Real_v smin   = -(signZdir * unplaced.fDz + point.z()) * invdir;

    //
    // Step 2: find distances for intersections with side planes.
    //

#ifdef VECGEOM_PLANESHELL
    // If disttoplanes is such that smin < dist < smax, then distance=disttoplanes
    Real_v disttoplanes = unplaced.GetPlanes()->DistanceToIn(point, dir, smin, smax);
    if (!done) distance = disttoplanes;

#else

    // here for VECGEOM_PLANESHELL_DISABLE

    // loop over side planes - find pdist,Comp for each side plane
    Real_v pdist[4], comp[4], vdist[4];
    // EvaluateTrack<Real_v>(unplaced, point, dir, pdist, comp, vdist);

    // auto-vectorizable part of loop
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    for (unsigned int i = 0; i < 4; ++i) {
      // Note: normal vector is pointing outside the volume (convention), therefore
      // pdist>0 if point is outside  and  pdist<0 means inside
      pdist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;

      // Comp is projection of dir over the normal vector of side plane, hence
      // Comp > 0 if pointing ~same direction as normal and Comp<0 if ~opposite to normal
      comp[i] = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();

      vdist[i] = -pdist[i] / NonZero(comp[i]);
    }

    // check special cases
    for (int i = 0; i < 4; ++i) {
      // points fully outside a plane and moving away or parallel to that plane
      done = done || (pdist[i] > Real_v(MakePlusTolerant<true>(0.)) && comp[i] >= Real_v(0.));
      // points at a plane surface and exiting
      done = done || (pdist[i] > Real_v(MakeMinusTolerant<true>(0.)) && comp[i] > Real_v(0.));
    }
    // if all particles moving away, we're done
    if (done) return;

    // this part does not auto-vectorize
    for (unsigned int i = 0; i < 4; ++i) {
      // if outside and moving away, return infinity
      bool posPoint = pdist[i] > Real_v(MakeMinusTolerant<true>(0.));
      bool posDir   = comp[i] > 0;

      // check if trajectory will intercept plane within current range (smin,smax), otherwise track misses shape
      bool interceptFromInside  = (!posPoint && posDir);
      bool interceptFromOutside = (posPoint && !posDir);

      //.. If dist is such that smin < dist < smax, then adjust either smin or smax
      if (interceptFromInside && vdist[i] < smax) smax = vdist[i];
      if (interceptFromOutside && vdist[i] > smin) smin = vdist[i];
    }

    if (!done && smin <= smax) distance = smin;
    if (distance < Real_v(MakeMinusTolerant<true>(0.0))) distance = Real_v(-1.);
#endif
  }

  /// @brief Compute the first exit distance from inside or on the trapezoid surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @param stepMax Unused by this implementation.
  /// @param[out] distance Exit distance, zero for an immediate tolerated exit, or `-1` for outside input.
  ///
  /// @details The z planes and side planes are checked with distance and
  /// direction tolerances. A start already outside any limiting plane reports
  /// `-1`; a tolerated surface start moving outward reports zero.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    const Real_v distanceTolerance  = kToleranceDist<Real_v>;
    const Real_v directionTolerance = kToleranceStrict<Real_v> * Real_v(unplaced.fInvProjectionScale);

    // step 0: if point is outside any plane --> return -1, otherwise initialize at Infinity
    bool outside = Abs(point.z()) > unplaced.fDz + distanceTolerance;
    distance     = InfinityLength<Real_v>();

    //
    // Step 1: find range of distances along dir between Z-planes (smin, smax)
    //

    if (Abs(dir.z()) > directionTolerance) {
      Real_v distz = (Sign(dir.z()) * unplaced.fDz - point.z()) / NonZero(dir.z());
      if (Abs(Abs(point.z()) - unplaced.fDz) <= distanceTolerance && point.z() * dir.z() > Real_v(0.)) {
        distance = Real_v(0.);
      } else if (distz >= -distanceTolerance && distz < distance) {
        distance = Max(distz, Real_v(0.));
      }
    }

    //
    // Step 2: find distances for intersections with side planes.
    //

#if defined(VECGEOM_PLANESHELL) && defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERAL_ACCELERATION)
    unplaced.GetPlanes()->DistanceToOut(point, dir, distanceTolerance, directionTolerance, outside, distance);
#else
    Real_v pdist[4], proj[4], vdist[4];
    EvaluateTrack<Real_v>(unplaced, point, dir, pdist, proj, vdist);

    for (unsigned int i = 0; i < 4; ++i) {
      outside |= pdist[i] > distanceTolerance;
      if (pdist[i] < -distanceTolerance) {
        if (proj[i] > directionTolerance && vdist[i] < distance) distance = vdist[i];
      } else if (pdist[i] <= distanceTolerance && proj[i] > directionTolerance) {
        distance = Real_v(0.);
      }
    }
#endif

    if (outside) distance = Real_v(-1.);
  }

  /// @brief Compute safety from an exterior point to the trapezoid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data.
  /// @param point Local point.
  /// @param[out] safety Conservative distance to the nearest entry boundary.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety = Abs(point.z()) - unplaced.fDz;

#ifdef VECGEOM_PLANESHELL
    // Get safety over side planes
    unplaced.GetPlanes()->SafetyToIn(point, safety);
#else
    // Loop over side planes
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    }

    Real_v safmax = Max(Max(dist[0], dist[1]), Max(dist[2], dist[3]));
    if (safmax > safety) safety = safmax;
#endif
  }

  /// @brief Compute safety from an interior point to leave the trapezoid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data.
  /// @param point Local point.
  /// @param[out] safety Conservative distance to the nearest exit boundary; negative for outside points.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    // If point is outside (wrong-side) --> safety to negative value
    safety = unplaced.fDz - Abs(point.z());

#ifdef VECGEOM_PLANESHELL
    // Get safety over side planes
    unplaced.GetPlanes()->SafetyToOut(point, safety);
#else
    // Loop over side planes
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();

    // auto-vectorizable loop
    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = -(fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD);
    }

    Real_v safmin = Min(Min(dist[0], dist[1]), Min(dist[2], dist[3]));
    if (safmin < safety) safety = safmin;
#endif
  }

  /// @brief Compute a trapezoid surface normal.
  /// @tparam Real_v Floating-point scalar type.
  /// @param unplaced Trapezoid data.
  /// @param point Local point.
  /// @param[out] valid Set when @p point is close enough to a limiting surface.
  /// @return Outward normal for the closest limiting plane, or a summed edge/corner normal.
  ///
  /// @details Side and z planes are compared using the local surface tolerance.
  /// Points on multiple surfaces return the normalized sum of the contributing
  /// outward normals.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {

    VECGEOM_CONST Precision delta = 1000. * kTolerance;
    Vector3D<Real_v> normal, cornerNormal;
    Real_v safety = InfinityLength<Real_v>();
    bool edge     = false;

#ifdef VECGEOM_PLANESHELL
    // Get normal from side planes -- PlaneShell case
    safety = unplaced.GetPlanes()->NormalKernel(point, normal, edge);

#else
    // Loop over side planes; reuse the accumulator declared above.
    unsigned char surfaces = 0;
    // vectorizable loop
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = Abs(fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD);
      // If closest update normal
      if (dist[i] < safety) {
        normal = unplaced.normals[i];
        safety = dist[i];
      }
      // If on surface add to separate vector
      if (dist[i] < kTolerance) {
        surfaces++;
        cornerNormal += unplaced.normals[i];
      }
    }

    if (surfaces > 1) {
      // The point is on the edge - do not normalize the vector
      normal = cornerNormal;
      edge   = true;
    }
#endif

    // check if normal is valid w.r.t. z-planes, and define normals based on safety (see above)
    Real_v safz = Abs(Abs(point[2]) - unplaced.fDz);
    if (edge && safz < kTolerance) {
      // The point is on a corner
      normal += Vector3D<Real_v>(0., 0., Sign(point.z()));
    } else {
      if (safz < safety) {
        normal.Set(0., 0., Sign(point.z()));
        safety = safz;
      }
    }
    valid = Abs(safety) <= delta;
    // returned vector must be normalized
    normal.Normalize();
    return normal;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
