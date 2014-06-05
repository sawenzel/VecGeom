/*
 * @file TrapezoidImplementation.h
 * @author Guilherme Lima (lima at fnal dot gov)
 *
 * 2014-05-14 G.Lima - Created using interface from Johannes's parallelepiped example and UTrap's implementation code
 */

#ifndef VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_

#include "base/Global.h"

//#include "base/Transformation3D.h"
// #include "volumes/kernel/BoxImplementation.h"
// #include "volumes/kernel/GenericKernels.h"
#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedTrapezoid.h"

//#include <stdio.h>
#include <sstream>

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
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside);

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

/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation<transCodeT, rotCodeT>::Transform(
    UnplacedTrapezoid const &unplaced,
    Vector3D<typename Backend::precision_v> &point) {
  point[1] -= unplaced.GetTanThetaSinPhi()*point[2];
  point[0] -= unplaced.GetTanThetaCosPhi()*point[2]
              + unplaced.GetTanAlpha()*point[1];
}
*/

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::UnplacedContains(
  UnplacedTrapezoid const &unplaced,
  Vector3D<typename Backend::precision_v> const &point,
  typename Backend::bool_v &inside) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Bool_t done(false);
    inside = Backend::kTrue;

    // point is outside if beyond trapezoid's z-length
    auto test = Abs(point.z()) >= unplaced.GetDz();
    MaskedAssign(test, Backend::kFalse, &inside);

    // if all points are outside, we're done
    done |= test;
    if (done == Backend::kTrue) return;

    // next check where points are w.r.t. each side plane
    Float_t Dist[4];
    TrapSidePlane const* planes = unplaced.GetPlanes();
    for (unsigned int i = 0; i < 4; ++i) {
      Dist[i] = planes[i].fA * point.x() + planes[i].fB * point.y()
        + planes[i].fC * point.z() + planes[i].fD;

      // is it outside of this side plane?
      test = (Dist[i] > Backend::kZero);  // no need to check (!done) here
      MaskedAssign(test, Backend::kFalse, &inside);

      // if all points are outside, we're done
      done |= test;
      if (done == Backend::kTrue) return;
    }

    // at this point, all points outside volume were tagged
    // next, check which points are in surface
    test = (!done) && ( Abs(point.z()) >= (unplaced.GetDz() - kHalfTolerance) );
    MaskedAssign(test, EInside::kSurface, &inside);
    done |= test;
    if (done == Backend::kTrue) return;

    for (unsigned int i = 0; i < 4; ++i) {
      test = (!done) && (Dist[i] > -kHalfTolerance);
      MaskedAssign(test, EInside::kSurface, &inside);
      done |= test;
      if (done == Backend::kTrue) return;
    }

    // at this point, all points outside or at surface were also tagged
    return;
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
  UnplacedContains<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedTrapezoid const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  Vector3D<typename Backend::precision_v> localPoint =
    transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedInside<Backend>(unplaced, localPoint, inside);
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

  distance = Backend::kZero;    // default returned value
  Float_t infinity = kInfinity;

  //
  // Step 1: find range of distances along dir between Z-planes (smin, smax)
  //

  // convenience variables for direction pointing to +z or -z.
  // Note that both posZdir and NegZdir may be false, if dir.z() is zero!
  Bool_t posZdir  = dir.z() > Backend::kZero;
  Bool_t negZdir = dir.z() < Backend::kZero;
  Float_t zdirSign = Backend::kOne;  // z-direction
  MaskedAssign( dir.z()<Backend::kZero, -Backend::kOne, &zdirSign);

  Float_t max;
  Float_t dirFactor = Backend::kOne / dir.z();     // convert distances from z to dir
  max = zdirSign*unplaced.GetDz() - point.z();     // z-dist to farthest z-plane

  // step 1.a) input particle is moving away --> return infinity

  // check if moving away towards +z
  Bool_t test = posZdir && max < kHalfTolerance;
  MaskedAssign( test, kInfinity, &distance );

  // check if moving away towards -z
  test = negZdir && max > -kHalfTolerance;
  MaskedAssign( test, kInfinity, &distance );

  // if all particles moving away, we're done
  Bool_t done( distance == infinity );
  if (done == Backend::kTrue ) return;

  // Step 1.b) General case:
  //   smax,smin are range of distances within z-range, taking direction into account.
  //   smin<smax - smax is positive, but smin may be either positive or negative
  Float_t smax = max * dirFactor;
  Float_t smin = (-zdirSign*unplaced.GetDz() - point.z())*dirFactor;

  // Step 1.c) special case: if dir is perpendicular to z-axis...
  test = (!posZdir) && (!negZdir);

  // ... and within z-range, then smin=0, smax=infinity for now
  Bool_t zrange = Abs(point.z()) < unplaced.GetDz() - kHalfTolerance;
  MaskedAssign( test && zrange, Backend::kZero, &smin );
  MaskedAssign( test && zrange,      kInfinity, &smax );

  // ... or out of z-range, then trajectory will not intercept volume
  MaskedAssign( test && !zrange, kInfinity, &distance );

  assert( (smin<smax) && "TrapezoidImplementation: smin<smax problem in DistanceToIn().");

  //
  // Step 2: find distances for intersections with side planes.
  //   If dist is such that smin < dist < smax, then adjust either smin or smax.
  //

  Float_t pdist, Comp;
  TrapSidePlane const* fPlanes = unplaced.GetPlanes();

  // loop over side planes - find pdist,Comp for each side plane
  for (unsigned int i = 0; i < 4; i++) {

    // Note: normal vector is pointing outside the volume (convention), therefore
    // pdist>0 if point is outside  and  pdist<0 means inside
    pdist = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y()
      + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    Bool_t posPoint = pdist >= -kHalfTolerance;

    // Comp is projection of dir over the normal vector of side plane, hence
    // Comp > 0 if pointing ~same direction as normal and Comp<0 if ~opposite to normal
    Comp = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();
    Bool_t posDir = Comp >= 0;

    // discard the ones moving away from this plane
    MaskedAssign( posPoint && posDir,   kInfinity, &distance  );

    // in original UTrap algorithm, the cases above are returned immediately
    // MaskedAssign( !done, distance == infinity, &done );
    done |= (distance == infinity);
    if (done == Backend::kTrue ) return;

    Bool_t interceptFromOutside = (!posPoint && posDir);
    Bool_t interceptFromInside = (posPoint && !posDir);

    // check if trajectory will intercept plane within current range (smin,smax)
    Float_t vdist = -pdist/Comp;
    Bool_t intercept = (vdist>Backend::kZero); // equivalent to interceptFromInside||interceptFromOutside

    MaskedAssign( interceptFromOutside && vdist<smin, kInfinity, &distance );
    MaskedAssign( interceptFromInside  && vdist>smax, kInfinity, &distance );
    done |= (distance == infinity);
    if (done == Backend::kTrue ) return;

    Bool_t validVdist = (vdist>smin && vdist<smax);
    MaskedAssign( interceptFromOutside && validVdist, vdist, &smax );
    MaskedAssign( interceptFromInside  && validVdist, vdist, &smin );

    assert( (smin<smax) && "TrapezoidImplementation: smin<smax problem in DistanceToIn().");
  }

  // Checks in non z plane intersections ensure smin<smax
  MaskedAssign(!done && smin>=0, smin, &distance);
  MaskedAssign(!done && smin<0, Backend::kZero, &distance);
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

  // typename Backend::int_v side = EInside::kInside;
  distance = kInfinity;            // init to invalid value
  Float_t infinity(kInfinity);

  //
  // Step 1: find range of distances along dir between Z-planes (smin, smax)
  //

  // convenience variables for direction pointing to +z or -z.
  // Note that both posZdir and NegZdir may be false, if dir.z() is zero!
  Bool_t posZdir = dir.z() > Backend::kZero;
  Bool_t negZdir = dir.z() < Backend::kZero;
  Float_t zdirSign = Backend::kOne;  // z-direction
  MaskedAssign( negZdir, -Backend::kOne, &zdirSign);

  Float_t zdirFactor = Backend::kOne / dir.z();     // convert distances from z to dir
  Float_t max = zdirSign*unplaced.GetDz() - point.z();  // z-dist to farthest z-plane

  // check if moving away towards +z
  Bool_t test = posZdir && max <= kHalfTolerance;
  MaskedAssign( test, Backend::kZero, &distance );

  // check if moving away towards -z
  test = negZdir && max >= -kHalfTolerance;
  MaskedAssign( test, Backend::kZero, &distance );

  // if all particles moving away, we're done
  Bool_t done( distance == Backend::kZero );
  if (done == Backend::kTrue ) return;

  // Step 1.b) general case:
  MaskedAssign( !done,  max/dir.z(), &distance);

  // Step 1.c) special case: if dir is perpendicular to z-axis...
  MaskedAssign(!posZdir && !negZdir, kInfinity, &distance);

  //
  // Step 2: find distances for intersections with side planes. 
  //   If dist is such that smin < dist < smax, then adjust either smin or smax.
  //

  Float_t pdist, Comp;
  TrapSidePlane const* fPlanes = unplaced.GetPlanes();

  // loop over side planes - find pdist,Comp for each side plane
  for (unsigned int i = 0; i < 4; i++) {

    // Note: normal vector is pointing outside the volume (convention), therefore
    // pdist>0 if point is outside  and  pdist<0 means inside
    pdist = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y()
      + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    Bool_t inside = (pdist < -kHalfTolerance);

    // Comp is projection of dir over the normal vector of side plane, hence
    // Comp > 0 if pointing ~same direction as normal and Comp<0 if ~opposite to normal
    Comp = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();
    Bool_t posComp = Comp >= 0;

    test = (!inside && posComp);
    MaskedAssign( !done && test, Backend::kZero, &distance );
    done |= ( distance == Backend::kZero );
    if (done == Backend::kTrue ) return;

    test = inside && posComp;
    Float_t vdist = -pdist / Comp;
    MaskedAssign(!done && test && vdist<distance, vdist, &distance);
  }

  // check for problems
  test = ( distance == infinity );
  assert( test == Backend::kFalse && "Undefined side for valid surface normal to solid.");
  if(test!=Backend::kFalse) {
    std::ostringstream message;
    int oldprc = message.precision(16);
    message << "Undefined side for valid surface normal to solid."
            << std::endl
            << "Position:"  << std::endl << std::endl
            << "point.x = "  << point.x() << " mm" << std::endl
            << "point.y = "  << point.y() << " mm" << std::endl
            << "point.z = "  << point.z() << " mm" << std::endl << std::endl
            << "Direction:" << std::endl << std::endl
            << "dir.x = "  << dir.x() << std::endl
            << "dir.y = "  << dir.y() << std::endl
            << "dir.z = "  << dir.z() << std::endl << std::endl
            << "Proposed distance: " << distance <<"\n\n";
    message.precision(oldprc);
    std::cout<< "TrapezoidImplementation.h: Warning: "<< message.str().c_str() <<"\n";
  }
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedTrapezoid const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  safety = Abs(point.z()) - unplaced.GetDz();

  // Loop over side planes
  TrapSidePlane const* fPlanes = unplaced.GetPlanes();
  for (int i = 0; i < 4; ++i) {
    Float_t Dist = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y()
      + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    MaskedAssign( Dist>safety, Dist, &safety );
  }

  MaskedAssign( safety < 0, Backend::kZero, &safety );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void TrapezoidImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedTrapezoid const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Bool_t done(false);

  // If point is outside, set distance to zero
  safety = unplaced.GetDz() - Abs(point.z());
  MaskedAssign( safety<Backend::kZero, Backend::kZero, &safety );
  done |= safety==Backend::kZero;

  // If all test points are outside, we're done
  if ( done == Backend::kTrue ) return;

  // Loop over side planes
  TrapSidePlane const* fPlanes = unplaced.GetPlanes();
  Float_t Dist( Backend::kZero );
  for (int i = 0; i < 4; ++i) {
    Dist = -(fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y()
             + fPlanes[i].fC * point.z() + fPlanes[i].fD);
    MaskedAssign( !done && Dist>Backend::kZero && Dist < safety, Dist, &safety );
  }
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
