/*
 * @file TrapezoidImplementation.h
 * @author Guilherme Lima (lima at fnal dot gov)
 *
 * 2014-05-14 G.Lima - Created using interface from Johannes's parallelepiped example and UTrap's implementation code
 */

#ifndef VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTrapezoid.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct TrapezoidImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedTrapezoid const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedTrapezoid const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTrapezoid const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &masterPoint,
      typename Backend::inside_v &inside);

  template <typename Backend, bool ForInside>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void GenericKernelForContainsAndInside(
      UnplacedTrapezoid const &unplaced,
      Vector3D<typename Backend::precision_v> const &,
      typename Backend::bool_v &,
      typename Backend::bool_v &);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedTrapezoid const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTrapezoid const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTrapezoid const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTrapezoid const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

};


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
    UnplacedTrapezoid const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyInside,
    typename Backend::bool_v &completelyOutside) {

  // z-region
  completelyOutside = Abs(localPoint[2]) > MakePlusTolerant<ForInside>( unplaced.GetDz() );
  if ( Backend::early_returns && IsFull(completelyOutside) )  return;
  if (ForInside) {
    completelyInside = Abs(localPoint[2]) < MakeMinusTolerant<ForInside>( unplaced.GetDz() );
  }

  // next check where points are w.r.t. each side plane
  typename Backend::precision_v Dist[4];
  unplaced.GetPlanes()->DistanceToPoint(localPoint, Dist);

  for (unsigned int i = 0; i < 4; ++i) {
    // is it outside of this side plane?
    completelyOutside |= Dist[i] > MakePlusTolerant<ForInside>(0.);
    if ( Backend::early_returns && IsFull(completelyOutside) )  return;
    if(ForInside) {
      completelyInside &= Dist[i] < MakeMinusTolerant<ForInside>(0.);
    }
  }

  return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::UnplacedContains(
  UnplacedTrapezoid const &unplaced,
  Vector3D<typename Backend::precision_v> const &point,
  typename Backend::bool_v &inside) {

  typedef typename Backend::bool_v Bool_t;
  Bool_t unused;
  Bool_t outside;
  GenericKernelForContainsAndInside<Backend, false>(unplaced, point, unused, outside);
  inside=!outside;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedTrapezoid const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);

  typedef typename Backend::bool_v Bool_t;

  Bool_t unused, outside;
  GenericKernelForContainsAndInside<Backend,false>(unplaced, localPoint,
                                                   unused, outside);

  inside = !outside;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedTrapezoid const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &masterPoint,
    typename Backend::inside_v &inside) {

  typedef typename Backend::bool_v Bool_t;

  // convert from master to local coordinates
  Vector3D<typename Backend::precision_v> point =
      transformation.Transform<transCodeT, rotCodeT>(masterPoint);

  Bool_t fullyinside, fullyoutside;
  GenericKernelForContainsAndInside<Backend,true>(
      unplaced, point, fullyinside, fullyoutside);
  inside=EInside::kSurface;
  MaskedAssign(fullyinside,  EInside::kInside,  &inside);
  MaskedAssign(fullyoutside, EInside::kOutside, &inside);
}

////////////////////////////////////////////////////////////////////////////
//
// Calculate distance to shape from outside - return kInfinity if no intersection
//
// ALGORITHM:
// For each component, calculate pair of minimum and maximum intersection
// values for which the particle is in the extent of the shape
// - The smallest (MAX minimum) allowed distance of the pairs is intersect
//
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedTrapezoid const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &masterPoint,
    Vector3D<typename Backend::precision_v> const &masterDir,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Vector3D<Float_t> point =
      transformation.Transform<transCodeT, rotCodeT>(masterPoint);
  Vector3D<Float_t> dir =
      transformation.TransformDirection<rotCodeT>(masterDir);

  distance = kInfinity;    // set to early returned value
  Bool_t done = Backend::kFalse;

  //
  // Step 1: find range of distances along dir between Z-planes (smin, smax)
  //

  // convenience variables for direction pointing to +z or -z.
  // Note that both posZdir and NegZdir may be false, if dir.z() is zero!
  Bool_t posZdir  = dir.z() > 0.0;
  Bool_t negZdir  = dir.z() < 0.0;
  Float_t zdirSign = Backend::kOne;  // z-direction
  MaskedAssign( negZdir, -Backend::kOne, &zdirSign);

  Float_t max = zdirSign*unplaced.GetDz() - point.z();     // z-dist to farthest z-plane

  // step 1.a) input particle is moving away --> return infinity

  // check if moving away towards +z
  done |= ( posZdir && max < kHalfTolerance );

  // check if moving away towards -z
  done |= ( negZdir && max >-kHalfTolerance );

  // if all particles moving away, we're done
  if (IsFull(done)) return;

  // Step 1.b) General case:
  //   smax,smin are range of distances within z-range, taking direction into account.
  //   smin<smax - smax is positive, but smin may be either positive or negative
  Float_t dirFactor = Backend::kOne / dir.z();     // convert distances from z to dir
  Float_t smax = max * dirFactor;
  Float_t smin = (-zdirSign*unplaced.GetDz() - point.z())*dirFactor;

  // Step 1.c) special case: if dir is perpendicular to z-axis...
  Bool_t test = (!posZdir) && (!negZdir);

  // ... and out of z-range, then trajectory will not intercept volume
  Bool_t zrange = Abs(point.z()) < unplaced.GetDz() - kHalfTolerance;
  done |= ( test && !zrange );
  if (IsFull(done)) return;

  // ... or within z-range, then smin=0, smax=infinity for now
  // GL note: in my environment, smin=-inf and smax=inf before these lines
  MaskedAssign( test && zrange,       0.0, &smin );
  MaskedAssign( test && zrange, kInfinity, &smax );


  //
  // Step 2: find distances for intersections with side planes.
  //   If disttoplanes is such that smin < dist < smax, then distance=disttoplanes
  //

  Float_t disttoplanes = unplaced.GetPlanes()->DistanceToIn<Backend>(point, smin, dir);

  // save any misses from plane shell
  done |= (disttoplanes==kInfinity);

  // reconciliate side planes w/ z-planes
  done |= (disttoplanes > smax);
  if (IsFull(done)) return;


  // at this point we know there is a valid distance - start with the z-plane based one
  MaskedAssign( !done, smin, &distance);

  // and then take the one from the planar shell, if valid
  Bool_t hitplanarshell = (disttoplanes > smin) && (disttoplanes < smax);
  MaskedAssign( hitplanarshell, disttoplanes, &distance);

  // at last... negative distance means we're inside the volume -- set distance to zero
  MaskedAssign( distance<0, 0.0, &distance );
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedTrapezoid const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &dir,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  distance = Backend::kZero;     // early return value
  Bool_t done(Backend::kFalse);

  //
  // Step 1: find range of distances along dir between Z-planes (smin, smax)
  //

  // convenience variables for direction pointing to +z or -z.
  // Note that both posZdir and NegZdir may be false, if dir.z() is zero!
  Bool_t posZdir = dir.z() > 0.0;
  Bool_t negZdir = dir.z() < 0.0;
  Float_t zdirSign = Backend::kOne;  // z-direction
  MaskedAssign( negZdir, -Backend::kOne, &zdirSign);

  Float_t max = zdirSign*unplaced.GetDz() - point.z();  // z-dist to farthest z-plane

  // do we need this ????
  // check if moving away towards +z
  done |= (posZdir && max <= kHalfTolerance);
  // check if moving away towards -z
  done |= (negZdir && max >= -kHalfTolerance);

  // if all particles moving away, we're done
  if (IsFull(done) ) return;

  // Step 1.b) general case: assign distance to z plane
  distance = max/dir.z();

  //
  // Step 2: find distances for intersections with side planes.
  //

  Float_t disttoplanes = unplaced.GetPlanes()->DistanceToOut<Backend>(point, dir);
  Bool_t hitplanarshell = (disttoplanes < distance);
  MaskedAssign( hitplanarshell, disttoplanes, &distance);
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::SafetyToIn(
  UnplacedTrapezoid const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &masterPoint,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  // convert from master to local coordinates
  Vector3D<Float_t> point = transformation.Transform<transCodeT, rotCodeT>(masterPoint);

  safety = Abs(point.z()) - unplaced.GetDz();

  // Loop over side planes
  typename Backend::precision_v Dist[4];
  unplaced.GetPlanes()->DistanceToPoint(point, Dist);
  for (int i = 0; i < 4; ++i) {
    MaskedAssign( Dist[i]>safety, Dist[i], &safety );
  }

  MaskedAssign(safety<0, 0.0, &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedTrapezoid const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::bool_v Bool_t;

  Bool_t done(false);

  // If point is outside, set distance to zero
  safety = unplaced.GetDz() - Abs(point.z());
  MaskedAssign( safety<0.0, 0.0, &safety );
  done |= safety==0.0;

  // If all test points are outside, we're done
  if ( IsFull(done) ) return;

  // Loop over side planes
  typename Backend::precision_v Dist[4];
  Planes const* planes = unplaced.GetPlanes();
  planes->DistanceToPoint(point, Dist);
  for (int i = 0; i < 4; ++i) {
    MaskedAssign( !done && Dist[i]<0.0 && -Dist[i] < safety, -Dist[i], &safety );
  }
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
