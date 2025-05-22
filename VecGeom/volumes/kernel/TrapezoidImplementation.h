//===-- kernel/TrapezoidImplementation.h ----------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
///
/// \file   kernel/TrapezoidImplementation.h
/// \author Guilherme Lima (lima@fnal.gov)
/// \brief This file implements the algorithms for the trapezoid
///
/// Implementation details: initially based on USolids algorithms and vectorized types.
///
//===--------------------------------------------------------------------------===//
///
/// 140520  G. Lima   Created from USolids' UTrap algorithms
/// 160722  G. Lima   Revision + moving to new backend structure

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

struct TrapezoidImplementation {

  using PlacedShape_t    = PlacedTrapezoid;
  using UnplacedStruct_t = TrapezoidStruct<Precision>;
  using UnplacedVolume_t = UnplacedTrapezoid;
#ifndef VECGEOM_PLANESHELL
  using TrapSidePlane = TrapezoidStruct<Precision>::TrapSidePlane;
#endif

#ifndef VECGEOM_PLANESHELL
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void EvaluateTrack(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir, Real_v *pdist,
                                                                         Real_v *proj, Real_v *vdist)
  {
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
  }
#endif

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(unplaced, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v completelyInside(false), completelyOutside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(unplaced, point, completelyInside, completelyOutside);

    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    inside             = Inside_t(EInside::kSurface);
    vecCore__MaskedAssignFunc(inside, (InsideBool_v)completelyOutside, Inside_t(EInside::kOutside));
    vecCore__MaskedAssignFunc(inside, (InsideBool_v)completelyInside, Inside_t(EInside::kInside));

    return;
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &completelyInside,
      Bool_v &completelyOutside)
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
      // if (vecCore::EarlyReturnMaxLength(completelyOutside,1) && vecCore::MaskFull(completelyOutside)) return;
    }
#endif

    return;
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  // Calculate distance to shape from outside - return kInfLength if no
  // intersection.
  //
  // ALGORITHM: For each component (z-planes, side planes), calculate
  // pair of minimum (smin) and maximum (smax) intersection values for
  // which the particle is in the extent of the shape.  The point of
  // entrance (exit) is found by the largest smin (smallest smax).
  //
  //  If largest smin > smallest smax, the trajectory does not reach
  //  inside the shape.
  //
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using Bool_v = vecCore::Mask_v<Real_v>;
    distance     = kInfLength;

    //
    // Step 1: find range of distances along dir between Z-planes (smin, smax)
    //

    // step 1.a) input particle is moving away --> return infinity
    Real_v signZdir = Sign(dir.z());
    Real_v max      = signZdir * unplaced.fDz - point.z(); // z-dist to farthest z-plane

    // done = done || (dir.z()>0.0 && max < MakePlusTolerant<true>(0.));  // check if moving away towards +z
    // done = done || (dir.z()<0.0 && max > MakeMinusTolerant<true>(0.)); // check if moving away towards -z
    Bool_v done(signZdir * max < Real_v(MakePlusTolerant<true>(0.0))); // if outside + moving away towards +/-z

    // if all particles moving away, we're done
    if (vecCore::EarlyReturnMaxLength(done, 1) && vecCore::MaskFull(done)) return;

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
    vecCore::MaskedAssign(distance, !done, disttoplanes);

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
    if (vecCore::EarlyReturnMaxLength(done, 1) && vecCore::MaskFull(done)) return;

    // this part does not auto-vectorize
    for (unsigned int i = 0; i < 4; ++i) {
      // if outside and moving away, return infinity
      Bool_v posPoint = pdist[i] > Real_v(MakeMinusTolerant<true>(0.));
      Bool_v posDir   = comp[i] > 0;

      // check if trajectory will intercept plane within current range (smin,smax), otherwise track misses shape
      Bool_v interceptFromInside  = (!posPoint && posDir);
      Bool_v interceptFromOutside = (posPoint && !posDir);

      //.. If dist is such that smin < dist < smax, then adjust either smin or smax
      vecCore__MaskedAssignFunc(smax, interceptFromInside && vdist[i] < smax, vdist[i]);
      vecCore__MaskedAssignFunc(smin, interceptFromOutside && vdist[i] > smin, vdist[i]);
    }

    vecCore::MaskedAssign(distance, !done && smin <= smax, smin);
    vecCore__MaskedAssignFunc(distance, distance < Real_v(MakeMinusTolerant<true>(0.0)), Real_v(-1.));
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using Bool_v = vecCore::Mask_v<Real_v>;

    // step 0: if point is outside any plane --> return -1, otherwise initialize at Infinity
    Bool_v outside = Abs(point.z()) > MakePlusTolerant<true>(unplaced.fDz);
    distance       = vecCore::Blend(outside, Real_v(-1.0), InfinityLength<Real_v>());
    Bool_v done(outside);
    if (vecCore::EarlyReturnMaxLength(done, 1) && vecCore::MaskFull(done)) return;

    //
    // Step 1: find range of distances along dir between Z-planes (smin, smax)
    //

    Real_v distz = (Sign(dir.z()) * unplaced.fDz - point.z()) / NonZero(dir.z());
    vecCore__MaskedAssignFunc(distance, !done && Abs(dir.z()) /** maxXY*/ > kTolerance, distz);

    //
    // Step 2: find distances for intersections with side planes.
    //

#ifdef VECGEOM_PLANESHELL
    Real_v disttoplanes = unplaced.GetPlanes()->DistanceToOut(point, dir);
    vecCore::MaskedAssign(distance, disttoplanes < distance, disttoplanes);

#else
    //=== Here for VECGEOM_PLANESHELL_DISABLE

    // loop over side planes - find pdist,Proj for each side plane
    Real_v pdist[4], proj[4], vdist[4];
    // Real_v dist1(distance);
    // EvaluateTrack<Real_v>(unplaced, point, dir, pdist, proj, vdist);

    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    for (unsigned int i = 0; i < 4; ++i) {
      // Note: normal vector is pointing outside the volume (convention), therefore
      // pdist>0 if point is outside  and  pdist<0 means inside
      pdist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;

      // Proj is projection of dir over the normal vector of side plane, hence
      // Proj > 0 if pointing ~same direction as normal and Proj<0 if pointing ~opposite to normal
      proj[i] = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();

      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }

    // early return if point is outside of plane
    // for (unsigned int i = 0; i < 4; ++i) {
    //   done = done || (pdist[i] > MakePlusTolerant<true>(0.));
    // }
    // vecCore::MaskedAssign(dist1, done, Real_v(-1.0));
    // if (vecCore::EarlyReturnMaxLength(done,1) && vecCore::MaskFull(done)) return;

    // std::cerr<<"=== point="<< point <<", dir="<< dir <<", distance="<< distance <<"\n";
    for (unsigned int i = 0; i < 4; ++i) {
      // if track is pointing towards plane and vdist<distance, then distance=vdist
      // vecCore__MaskedAssignFunc(dist1, !done && proj[i] > 0.0 && vdist[i] < dist1, vdist[i]);
      vecCore__MaskedAssignFunc(distance, pdist[i] > MakePlusTolerant<true>(0.), Real_v(-1.0));
      vecCore__MaskedAssignFunc(distance, proj[i] * unplaced.fDz > kTolerance && -Sign(pdist[i]) * vdist[i] < distance,
                                -Sign(pdist[i]) * vdist[i]);
      // std::cerr<<"i="<< i <<", pdist="<< pdist[i] <<", proj="<< proj[i] <<", vdist="<< vdist[i] <<" --> dist="<<
      // dist1 <<", "<< distance <<"\n";
    }
#endif
  }

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

    // for (int i = 0; i < 4; ++i) {
    //   vecCore::MaskedAssign(safety, dist[i] > safety, dist[i]);
    // }
    Real_v safmax = Max(Max(dist[0], dist[1]), Max(dist[2], dist[3]));
    vecCore::MaskedAssign(safety, safmax > safety, safmax);
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    // If point is outside (wrong-side) --> safety to negative value
    safety = unplaced.fDz - Abs(point.z());

    // If all test points are outside, we're done
    // if (vecCore::EarlyReturnMaxLength(safety,1)) {
    //   if (vecCore::MaskFull(safety < kHalfTolerance)) return;
    // }

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

    // unvectorizable loop
    // for (int i = 0; i < 4; ++i) {
    //   vecCore::MaskedAssign(safety, dist[i] < safety, dist[i]);
    // }

    Real_v safmin = Min(Min(dist[0], dist[1]), Min(dist[2], dist[3]));
    vecCore::MaskedAssign(safety, safmin < safety, safmin);
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {

    VECGEOM_CONST Precision delta = 1000. * kTolerance;
    Vector3D<Real_v> normal(0.);
    Real_v safety = -InfinityLength<Real_v>();

#ifdef VECGEOM_PLANESHELL
    // Get normal from side planes -- PlaneShell case
    safety = unplaced.GetPlanes()->NormalKernel(point, normal);

#else
    // Loop over side planes

    // vectorizable loop
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = (fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD);
    }

    // non-vectorizable part
    for (int i = 0; i < 4; ++i) {
      Real_v saf_i = dist[i] - safety;

      // if more planes found as far (within tolerance) as the best one so far *and not fully inside*, add its normal
      vecCore__MaskedAssignFunc(normal, Abs(saf_i) < kHalfTolerance && dist[i] >= -kHalfTolerance,
                                normal + unplaced.normals[i]);

      // this one is farther than our previous one -- update safety and normal
      vecCore__MaskedAssignFunc(normal, saf_i > 0.0, unplaced.normals[i]);
      vecCore__MaskedAssignFunc(safety, saf_i > 0.0, dist[i]);
      // std::cerr<<"dist["<< i <<"]="<< dist[i] <<", saf_i="<< saf_i <<", safety="<< safety <<", normal="<< normal
      // <<"\n";
    }
#endif

    // check if normal is valid w.r.t. z-planes, and define normals based on safety (see above)
    Real_v safz(Sign(point[2]) * point[2] - unplaced.fDz);

    vecCore__MaskedAssignFunc(normal, Abs(safz - safety) < kHalfTolerance && safz >= -kHalfTolerance,
                              normal + Vector3D<Real_v>(0, 0, Sign(point.z())));

    vecCore__MaskedAssignFunc(normal, safz > safety && safz >= -kHalfTolerance,
                              Vector3D<Real_v>(0, 0, Sign(point.z())));
    vecCore::MaskedAssign(safety, safz > safety, safz);
    valid = Abs(safety) <= delta;
    // std::cerr<<"safz="<< safz <<", safety="<< safety <<", normal="<< normal <<", valid="<< valid <<"\n";

    // returned vector must be normalized
    if (normal.Mag2() > 1.0) normal.Normalize(); //??? check use of MaskedAssignFunc here!!!

    return normal;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
