//===-- kernel/BoxImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file BoxImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch), Sandro Wenzel (sandro.wenzel@cern.ch)

/// History notes:
/// 2013 - 2014: original development (abstracted kernels); Johannes and Sandro
/// Oct 2015: revision + moving to new backend structure (Sandro Wenzel)

#ifndef VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/BoxStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct BoxImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, BoxImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox;
template <typename T>
struct BoxStruct;
class UnplacedBox;

struct BoxImplementation {

  using PlacedShape_t    = PlacedBox;
  using UnplacedStruct_t = BoxStruct<Precision>;
  using UnplacedVolume_t = UnplacedBox;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> HalfSize(const UnplacedStruct_t &box)
  {
    return Vector3D<Real_v>(box.fDimensions[0], box.fDimensions[1], box.fDimensions[2]);
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &box,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    // in analogy to other shapes, surface points are considered inside
    inside = (point.Abs() - HalfSize<Real_v>(box)).Max() <= Real_v(0.0);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &box,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    Real_v dist = (point.Abs() - HalfSize<Real_v>(box)).Max();

    inside = vecCore::Blend(dist < Real_v(0.0), Inside_v(kInside), Inside_v(kOutside));
    vecCore__MaskedAssignFunc(inside, Abs(dist) < Real_v(kHalfTolerance), Inside_v(kSurface));
  }

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      Vector3D<Real_v> const &halfsize, Vector3D<Real_v> const &point, vecCore::Mask<Real_v> &completelyinside,
      vecCore::Mask<Real_v> &completelyoutside)
  {
    Real_v dist = (point.Abs() - halfsize).Max();

    if (ForInside) completelyinside = dist < Real_v(-kHalfTolerance);

    completelyoutside = dist > Real_v(kHalfTolerance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &box,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /* stepMax */, Real_v &distance)
  {
    const Vector3D<Real_v> invDir(Real_v(1.0) / NonZero(direction[0]), Real_v(1.0) / NonZero(direction[1]),
                                  Real_v(1.0) / NonZero(direction[2]));

    const Vector3D<Real_v> signDir(Sign(direction[0]), Sign(direction[1]), Sign(direction[2]));

    const Vector3D<Real_v> tempIn  = -signDir * box.fDimensions - point;
    const Vector3D<Real_v> tempOut = signDir * box.fDimensions - point;

    // add a check for point on exit surface
    const Real_v absOrthogOut = Abs((signDir * tempOut).Min());

    const Real_v distOut = (tempOut * invDir).Min();

    // distIn calculation
    distance = (tempIn * invDir).Max();

    vecCore__MaskedAssignFunc(
        distance, distance >= distOut || distOut <= Real_v(kHalfTolerance) || absOrthogOut <= Real_v(kHalfTolerance),
        InfinityLength<Real_v>());
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &box,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    const Vector3D<Real_v> invDir(Real_v(1.0) / NonZero(direction[0]), Real_v(1.0) / NonZero(direction[1]),
                                  Real_v(1.0) / NonZero(direction[2]));

    const Vector3D<Real_v> signDir(Sign(direction[0]), Sign(direction[1]), Sign(direction[2]));

    const Real_v safetyIn = (point.Abs() - HalfSize<Real_v>(box)).Max();

    const Vector3D<Real_v> tempOut = signDir * box.fDimensions - point;

    distance = (tempOut * invDir).Min();

    vecCore__MaskedAssignFunc(distance, safetyIn > Real_v(kHalfTolerance), Real_v(-1.0));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &box,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety = (point.Abs() - HalfSize<Real_v>(box)).Max();
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &box,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety = (HalfSize<Real_v>(box) - point.Abs()).Min();
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &box, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    //   In case a point is further than kHalfTolerance from a surface, set valid=false
    //   Must return a valid vector. (even if the point is not on the surface.)
    //
    //   On an edge or corner, provide an average normal of all facets within tolerance

    const Vector3D<Real_v> safety((point.Abs() - HalfSize<Real_v>(box)).Abs());
    const Real_v safmin = safety.Min();
    valid               = safmin < kHalfTolerance;

    Vector3D<Real_v> normal(0.);
    vecCore__MaskedAssignFunc(normal[0], safety[0] - safmin < kHalfTolerance, Sign(point[0]));
    vecCore__MaskedAssignFunc(normal[1], safety[1] - safmin < kHalfTolerance, Sign(point[1]));
    vecCore__MaskedAssignFunc(normal[2], safety[2] - safmin < kHalfTolerance, Sign(point[2]));
    if (normal.Mag2() > 1.0) normal.Normalize();

    return normal;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  // template <class Backend>
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool Intersect(Vector3D<Precision> const *corners, Vector3D<Precision> const &point,
                        Vector3D<Precision> const &ray, Precision /* t0 */, Precision /* t1 */)
  {
    // intersection algorithm 1 ( Amy Williams )
    Precision tmin, tmax, tymin, tymax, tzmin, tzmax;

    // IF THERE IS A STEPMAX; COULD ALSO CHECK SAFETIES
    Precision inverserayx = 1. / ray[0];
    Precision inverserayy = 1. / ray[1];

    // TODO: we should promote this to handle multiple boxes
    int sign[3];
    sign[0] = inverserayx < 0;
    sign[1] = inverserayy < 0;

    tmin  = (corners[sign[0]].x() - point.x()) * inverserayx;
    tmax  = (corners[1 - sign[0]].x() - point.x()) * inverserayx;
    tymin = (corners[sign[1]].y() - point.y()) * inverserayy;
    tymax = (corners[1 - sign[1]].y() - point.y()) * inverserayy;

    if ((tmin > tymax) || (tymin > tmax)) return false;

    Precision inverserayz = 1. / ray.z();
    sign[2]               = inverserayz < 0;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    tzmin = (corners[sign[2]].z() - point.z()) * inverserayz;
    tzmax = (corners[1 - sign[2]].z() - point.z()) * inverserayz;

    if ((tmin > tzmax) || (tzmin > tmax)) return false;
    // if ((tzmin > tmin)) tmin = tzmin;
    // if (tzmax < tmax) tmax   = tzmax;
    // return ((tmin < t1) && (tmax > t0));
    // std::cerr << "tmin " << tmin << " tmax " << tmax << "\n";
    return true;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  template <int signx, int signy, int signz>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
      //__attribute__((noinline))
      static Precision
      IntersectCached(Vector3D<Precision> const *corners, Vector3D<Precision> const &point,
                      Vector3D<Precision> const &inverseray, Precision t0, Precision t1)
  {
    // intersection algorithm 1 ( Amy Williams )

    // NOTE THE FASTEST VERSION IS STILL THE ORIGINAL IMPLEMENTATION

    Precision tmin, tmax, tymin, tymax, tzmin, tzmax;

    // TODO: we should promote this to handle multiple boxes
    // observation: we always compute sign and 1-sign; so we could do the assignment
    // to tmin and tmax in a masked assignment thereafter
    tmin  = (corners[signx].x() - point.x()) * inverseray.x();
    tmax  = (corners[1 - signx].x() - point.x()) * inverseray.x();
    tymin = (corners[signy].y() - point.y()) * inverseray.y();
    tymax = (corners[1 - signy].y() - point.y()) * inverseray.y();
    if ((tmin > tymax) || (tymin > tmax)) return InfinityLength<Precision>();

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    tzmin = (corners[signz].z() - point.z()) * inverseray.z();
    tzmax = (corners[1 - signz].z() - point.z()) * inverseray.z();

    if ((tmin > tzmax) || (tzmin > tmax)) return InfinityLength<Precision>(); // false
    if ((tzmin > tmin)) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    if (!((tmin < t1) && (tmax > t0))) return InfinityLength<Precision>();
    return tmin;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  template <typename Real_v, int signx, int signy, int signz>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v IntersectCachedKernel(
      Vector3D<Real_v> const *corners, Vector3D<Precision> const &point, Vector3D<Precision> const &inverseray,
      Precision t0, Precision t1)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v tmin  = (corners[signx].x() - point.x()) * inverseray.x();
    Real_v tmax  = (corners[1 - signx].x() - point.x()) * inverseray.x();
    Real_v tymin = (corners[signy].y() - point.y()) * inverseray.y();
    Real_v tymax = (corners[1 - signy].y() - point.y()) * inverseray.y();

    // do we need this condition ?
    Bool_v done = (tmin > tymax) || (tymin > tmax);
    if (vecCore::MaskFull(done)) return InfinityLength<Real_v>();
    // if((tmin > tymax) || (tymin > tmax))
    //     return vecgeom::kInfLength;

    // Not sure if this has to be maskedassignments
    tmin = Max(tmin, tymin);
    tmax = Min(tmax, tymax);

    Real_v tzmin = (corners[signz].z() - point.z()) * inverseray.z();
    Real_v tzmax = (corners[1 - signz].z() - point.z()) * inverseray.z();

    done |= (tmin > tzmax) || (tzmin > tmax);
    // if((tmin > tzmax) || (tzmin > tmax))
    //     return vecgeom::kInfLength; // false
    if (vecCore::MaskFull(done)) return InfinityLength<Real_v>();

    // not sure if this has to be maskedassignments
    tmin = Max(tmin, tzmin);
    tmax = Min(tmax, tzmax);

    done |= !((tmin < t1) && (tmax > t0));
    // if( ! ((tmin < t1) && (tmax > t0)) )
    //     return vecgeom::kInfLength;
    vecCore__MaskedAssignFunc(tmin, done, InfinityLength<Real_v>());
    return tmin;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  template <typename Real_v, typename basep>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v IntersectCachedKernel2(Vector3D<Real_v> const *corners,
                                                                                    Vector3D<basep> const &point,
                                                                                    Vector3D<basep> const &inverseray,
                                                                                    int signx, int signy, int signz,
                                                                                    basep t0, basep t1)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v tmin  = (corners[signx].x() - Real_v(point.x())) * inverseray.x();
    Real_v tymax = (corners[1 - signy].y() - Real_v(point.y())) * inverseray.y();
    Bool_v done  = tmin > tymax;
    if (vecCore::MaskFull(done)) return InfinityLength<Real_v>();

    Real_v tmax  = (corners[1 - signx].x() - Real_v(point.x())) * inverseray.x();
    Real_v tymin = (corners[signy].y() - Real_v(point.y())) * inverseray.y();

    // do we need this condition ?
    done |= (tymin > tmax);
    if (vecCore::MaskFull(done)) return InfinityLength<Real_v>();

    // if((tmin > tymax) || (tymin > tmax))
    //     return vecgeom::kInfLength;

    // Not sure if this has to be maskedassignments
    tmin = Max(tmin, tymin);
    tmax = Min(tmax, tymax);

    Real_v tzmin = (corners[signz].z() - point.z()) * inverseray.z();
    Real_v tzmax = (corners[1 - signz].z() - point.z()) * inverseray.z();

    done |= (Real_v(tmin) > Real_v(tzmax)) || (Real_v(tzmin) > Real_v(tmax));
    // if((tmin > tzmax) || (tzmin > tmax))
    //     return vecgeom::kInfLength; // false
    if (vecCore::MaskFull(done)) return InfinityLength<Real_v>();

    // not sure if this has to be maskedassignments
    tmin = Max(tmin, tzmin);
    tmax = Min(tmax, tzmax);

    done |= !((tmin <= Real_v(t1 + kTolerance)) && (tmax > Real_v(t0 - kTolerance)));
    // if( ! ((tmin < t1) && (tmax > t0)) )
    //     return vecgeom::kInfLength;
    vecCore__MaskedAssignFunc(tmin, done, InfinityLength<Real_v>());
    return tmin;
  }

  // an algorithm to test for intersection against many boxes but just one ray;
  // in this case, the inverse ray is cached outside and directly given here as input
  // we could then further specialize this function to the direction of the ray
  // because also the sign[] variables and hence the branches are predefined

  // one could do: template <class Backend, int sign0, int sign1, int sign2>
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Precision IntersectMultiple(Vector3D<Real_v> const lowercorners,
                                                                                  Vector3D<Real_v> const uppercorners,
                                                                                  Vector3D<Precision> const &point,
                                                                                  Vector3D<Precision> const &inverseray,
                                                                                  Precision t0, Precision t1)
  {
    // intersection algorithm 1 ( Amy Williams )

    typedef Real_v Float_t;

    Float_t tmin, tmax, tymin, tymax, tzmin, tzmax;
    // IF THERE IS A STEPMAX; COULD ALSO CHECK SAFETIES

    // TODO: we should promote this to handle multiple boxes
    // we might need to have an Index type

    // int sign[3];
    Float_t sign[3]; // this also exists
    sign[0] = inverseray.x() < 0;
    sign[1] = inverseray.y() < 0;

    // observation: we always compute sign and 1-sign; so we could do the assignment
    // to tmin and tmax in a masked assignment thereafter

    // tmin =  (corners[(int)sign[0]].x()   -point.x())*inverserayx;
    // tmax =  (corners[(int)(1-sign[0])].x() -point.x())*inverserayx;
    // tymin = (corners[(int)(sign[1])].y()   -point.y())*inverserayy;
    // tymax = (corners[(int)(1-sign[1])].y() -point.y())*inverserayy;

    Precision x0 = (lowercorners.x() - point.x()) * inverseray.x();
    Precision x1 = (uppercorners.x() - point.x()) * inverseray.x();
    Precision y0 = (lowercorners.y() - point.y()) * inverseray.y();
    Precision y1 = (uppercorners.y() - point.y()) * inverseray.y();
    // could we do this using multiplications?
    //    tmin =   !sign[0] ?  x0 : x1;
    //    tmax =   sign[0] ? x0 : x1;
    //    tymin =  !sign[1] ?  y0 : y1;
    //    tymax =  sign[1] ? y0 : y1;

    // could completely get rid of this ? because the sign is determined by the outside ray

    tmin  = (1 - sign[0]) * x0 + sign[0] * x1;
    tmax  = sign[0] * x0 + (1 - sign[0]) * x1;
    tymin = (1 - sign[1]) * y0 + sign[1] * y1;
    tymax = sign[1] * y0 + (1 - sign[1]) * y1;

    // tmax =  (corners[(int)(1-sign[0])].x() -point.x())*inverserayx;
    // tymin = (corners[(int)(sign[1])].y()   -point.y())*inverserayy;
    // tymax = (corners[(int)(1-sign[1])].y() -point.y())*inverserayy;

    if ((tmin > tymax) || (tymin > tmax)) return InfinityLength<Precision>();

    //  Precision inverserayz = 1./ray.z();
    sign[2] = inverseray.z() < 0;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    //
    // tzmin = (lowercorners[(int) sign[2]].z()   -point.z())*inverseray.z();
    // tzmax = (uppercorners[(int)(1-sign[2])].z() -point.z())*inverseray.z();

    if ((tmin > tzmax) || (tzmin > tmax)) return InfinityLength<Precision>(); // false
    if ((tzmin > tmin)) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    if (!((tmin < t1) && (tmax > t0))) return InfinityLength<Precision>();
    // std::cerr << "tmin " << tmin << " tmax " << tmax << "\n";
    // return true;
    return tmin;
  }
}; // End struct BoxImplementation

struct ABBoxImplementation {

  // a contains kernel to be used with aligned bounding boxes
  // scalar and vector modes (aka backend) for boxes but only single points
  // should be useful to test one point against many bounding boxes
  // TODO: check if this can be unified with the normal generic box kernel
  template <typename Real_v, typename Bool_v = typename vecCore::Mask_v<Real_v>>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ABBoxContainsKernel(Vector3D<Real_v> const &lowercorner,
                                                                               Vector3D<Real_v> const &uppercorner,
                                                                               Vector3D<Precision> const &point,
                                                                               Bool_v &inside)
  {

    inside = lowercorner.x() < Real_v(point.x());
    inside &= uppercorner.x() > Real_v(point.x());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.y() < Real_v(point.y());
    inside &= uppercorner.y() > Real_v(point.y());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.z() < Real_v(point.z());
    inside &= uppercorner.z() > Real_v(point.z());
  }

  // playing with a kernel that can do multi-box - single particle; multi-box -- multi-particle, single-box --
  // multi-particle
  template <typename T1, typename T2, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ABBoxContainsKernelGeneric(Vector3D<T1> const &lowercorner,
                                                                                      Vector3D<T1> const &uppercorner,
                                                                                      Vector3D<T2> const &point,
                                                                                      Bool_v &inside)
  {
    inside = lowercorner.x() < T1(point.x());
    inside &= uppercorner.x() > T1(point.x());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.y() < T1(point.y());
    inside &= uppercorner.y() > T1(point.y());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.z() < T1(point.z());
    inside &= uppercorner.z() > T1(point.z());
  }

  // safety square for Bounding boxes
  // generic kernel treating one track and one or multiple boxes
  // in case a point is inside a box a squared value
  // is returned but given an overall negative sign
  template <typename Real_v, typename Real_s = typename vecCore::TypeTraits<Real_v>::ScalarType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v ABBoxSafetySqr(Vector3D<Real_v> const &lowercorner,
                                                                            Vector3D<Real_v> const &uppercorner,
                                                                            Vector3D<Real_s> const &point)
  {

    using Vector3D_v = Vector3D<Real_v>;
    using Bool_v     = vecCore::Mask_v<Real_v>;

    const Vector3D_v kHalf(Real_v(static_cast<Real_s>(0.5)));
    const Vector3D_v origin((uppercorner + lowercorner) * kHalf);
    const Vector3D_v delta((uppercorner - lowercorner) * kHalf);
    // promote scalar point to vector point
    const Vector3D_v promotedpoint(Real_v(point.x()), Real_v(point.y()), Real_v(point.z()));

    // it would be nicer to have a standalone Abs function taking Vector3D as input
    const Vector3D_v safety = ((promotedpoint - origin).Abs()) - delta;
    const Bool_v outsidex   = safety.x() > Real_s(0.);
    const Bool_v outsidey   = safety.y() > Real_s(0.);
    const Bool_v outsidez   = safety.z() > Real_s(0.);

    Real_v runningsafetysqr(0.);                  // safety squared from outside
    Real_v runningmax(-InfinityLength<Real_v>()); // relevant for safety when we are inside

    // loop over dimensions manually unrolled
    // treat x dim
    {
      // this will be much simplified with operator notation
      Real_v tmp(0.);
      vecCore__MaskedAssignFunc(tmp, outsidex, safety.x() * safety.x());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.x());
    }

    // treat y dim
    {
      Real_v tmp(0.);
      vecCore__MaskedAssignFunc(tmp, outsidey, safety.y() * safety.y());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.y());
    }

    // treat z dim
    {
      Real_v tmp(0.);
      vecCore__MaskedAssignFunc(tmp, outsidez, safety.z() * safety.z());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.z());
    }

    Bool_v inside = !(outsidex || outsidey || outsidez);
    if (!vecCore::MaskEmpty(inside)) vecCore__MaskedAssignFunc(runningsafetysqr, inside, -runningmax * runningmax);
    return runningsafetysqr;
  }

  // safety square for Bounding boxes, returning the squared range for any point in the box
  // generic kernel treating one track and one or multiple boxes
  // in case a point is inside a box a squared value
  // is returned but given an overall negative sign
  template <typename Real_v, typename Real_s = typename vecCore::TypeTraits<Real_v>::ScalarType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v ABBoxSafetyRangeSqr(Vector3D<Real_v> const &lowercorner,
                                                                                 Vector3D<Real_v> const &uppercorner,
                                                                                 Vector3D<Real_s> const &point,
                                                                                 Real_v &safetymaxsqr)
  {

    using Vector3D_v = Vector3D<Real_v>;
    using Bool_v     = vecCore::Mask_v<Real_v>;

    const Vector3D_v kHalf(Real_v(static_cast<Real_s>(0.5)));
    const Vector3D_v origin((uppercorner + lowercorner) * kHalf);
    const Vector3D_v delta((uppercorner - lowercorner) * kHalf);
    // promote scalar point to vector point
    const Vector3D_v promotedpoint(Real_v(point.x()), Real_v(point.y()), Real_v(point.z()));

    // it would be nicer to have a standalone Abs function taking Vector3D as input
    const Vector3D_v safety  = ((promotedpoint - origin).Abs()) - delta;
    const Vector3D_v safetyp = ((promotedpoint - origin).Abs()) + delta;
    const Bool_v outsidex    = safety.x() > Real_s(0.);
    const Bool_v outsidey    = safety.y() > Real_s(0.);
    const Bool_v outsidez    = safety.z() > Real_s(0.);

    Real_v runningsafetysqr(0.);                  // safety squared from outside
    safetymaxsqr = safetyp.Mag2();                // safetymax squared from outside
    Real_v runningmax(-InfinityLength<Real_v>()); // relevant for safety when we are inside

    // loop over dimensions manually unrolled
    // treat x dim
    {
      // this will be much simplified with operator notation
      Real_v tmp(0.);
      vecCore__MaskedAssignFunc(tmp, outsidex, safety.x() * safety.x());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.x());
      //      vecCore__MaskedAssignFunc(tmp, outsidex, safetyp.x() * safetyp.x());
      //      safetymaxsqr += tmp;
    }

    // treat y dim
    {
      Real_v tmp(0.);
      vecCore__MaskedAssignFunc(tmp, outsidey, safety.y() * safety.y());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.y());
      //      vecCore__MaskedAssignFunc(tmp, outsidey, safetyp.y() * safetyp.y());
      //      safetymaxsqr += tmp;
    }

    // treat z dim
    {
      Real_v tmp(0.);
      vecCore__MaskedAssignFunc(tmp, outsidez, safety.z() * safety.z());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.z());
      //      vecCore__MaskedAssignFunc(tmp, outsidez, safetyp.z() * safetyp.z());
      //      safetymaxsqr += tmp;
    }

    Bool_v inside = !(outsidex || outsidey || outsidez);
    if (!vecCore::MaskEmpty(inside)) vecCore__MaskedAssignFunc(runningsafetysqr, inside, -runningmax * runningmax);
    return runningsafetysqr;
  }

}; // end aligned bounding box struct
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
