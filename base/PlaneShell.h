/// \file PlaneShell.h
/// \author Guilherme Lima (lima at fnal dot gov)

#ifndef VECGEOM_BASE_SIDEPLANES_H_
#define VECGEOM_BASE_SIDEPLANES_H_

#include "base/Global.h"
#include "volumes/kernel/GenericKernels.h"

//namespace vecgeom_cuda { template <typename Backend, int N> class PlaneShell; }
#include "backend/Backend.h"

namespace VECGEOM_NAMESPACE {

/**
 * @brief Uses SoA layout to store arrays of N (plane) parameters,
 *        representing a set of planes defining a volume or shape.
 *
 * For some volumes, e.g. trapezoid, when two of the planes are
 * parallel, they should be set perpendicular to the Z-axis, and then
 * the inside/outside calculations become trivial.  Therefore those planes
 * should NOT be included in this class.
 *
 * @details If vector acceleration is enabled, the scalar template
 *        instantiation will use vector instructions for operations
 *        when possible.
 */

//template <int N, typename Type>
template <int N, typename Type>
struct PlaneShell {

  // Using a SOA-like data structure for vectorization
  Precision fA[N];
  Precision fB[N];
  Precision fC[N];
  Precision fD[N];

public:

  /**
   * Initializes the SOA with existing data arrays, performing no allocation.
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell(Precision *const a, Precision *const b, Precision *const c, Precision *const d)
  {
    memcpy( &(this->fA), a, N*sizeof(Type) );
    memcpy( &(this->fB), b, N*sizeof(Type) );
    memcpy( &(this->fC), c, N*sizeof(Type) );
    memcpy( &(this->fD), d, N*sizeof(Type) );
  }


  /**
   * Initializes the SOA with a fixed size, allocating an aligned array for each
   * coordinate of the specified size.
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell()
  {
    memset( &(this->fA), 0, N*sizeof(Type) );
    memset( &(this->fB), 0, N*sizeof(Type) );
    memset( &(this->fC), 0, N*sizeof(Type) );
    memset( &(this->fD), 0, N*sizeof(Type) );
  }


  /**
   * Copy constructor
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell(PlaneShell const &other) {
    memcpy( &(this->fA), &(other->fA), N*sizeof(Type) );
    memcpy( &(this->fB), &(other->fB), N*sizeof(Type) );
    memcpy( &(this->fC), &(other->fC), N*sizeof(Type) );
    memcpy( &(this->fD), &(other->fD), N*sizeof(Type) );
  }


  /**
   * assignment operator
   */
  VECGEOM_CUDA_HEADER_BOTH
    PlaneShell& operator=(PlaneShell const &other) {
      memcpy( this->fA, other.fA, N*sizeof(Type) );
      memcpy( this->fB, other.fB, N*sizeof(Type) );
      memcpy( this->fC, other.fC, N*sizeof(Type) );
      memcpy( this->fD, other.fD, N*sizeof(Type) );
      return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
    void Set(int i, Precision a, Precision b, Precision c, Precision d) {
    fA[i] = a;
    fB[i] = b;
    fC[i] = c;
    fD[i] = d;
  }

  VECGEOM_CUDA_HEADER_BOTH
  unsigned int size() {
    return N;
  }

  VECGEOM_CUDA_HEADER_BOTH
  ~PlaneShell() { }

  /// \return the distance from point to each plane.  The type returned is float, double, or various SIMD vector types.
  /// Distances are negative (positive) for points in same (opposite) side from plane as the normal vector.
  template<typename Type2>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DistanceToPoint(Vector3D<Type2> const& point, Type2* distances) const {
    for(int i=0; i<N; ++i) {
      distances[i] = this->fA[i]*point.x() + this->fB[i]*point.y() + this->fC[i]*point.z() + this->fD[i];
    }
  }

  /// \return the projection of a (Vector3D) direction into each plane's normal vector.
  /// The type returned is float, double, or various SIMD vector types.
  template<typename Type2>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void ProjectionToNormal(Vector3D<Type2> const& dir, Type2* projection) const {
    for(int i=0; i<N; ++i) {
      projection[i] = this->fA[i]*dir.x() + this->fB[i]*dir.y() + this->fC[i]*dir.z();
    }
  }


  template <typename Backend, bool ForInside>
  VECGEOM_CUDA_HEADER_BOTH
  void GenericKernelForContainsAndInside(
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &completelyInside,
      typename Backend::bool_v &completelyOutside ) const {

    // auto-vectorizable loop for Backend==scalar
    typedef typename Backend::precision_v Float_t;
    Float_t dist[N];
    for(unsigned int i=0; i<N; ++i) {
      dist[i] = this->fA[i]*point.x() + this->fB[i]*point.y()
              + this->fC[i]*point.z() + this->fD[i];
    }

    // analysis loop - not auto-vectorizable
    for(unsigned int i=0; i<N; ++i) {
      // is it outside of this side plane?
      completelyOutside |= dist[i] > MakePlusTolerant<ForInside>(0.);
      if ( IsFull(completelyOutside) )  return;
      if ( ForInside ) {
        completelyInside &= dist[i] < MakeMinusTolerant<ForInside>(0.);
      }
    }
  }


  /// \return the distance to the planar shell when the point is located outside.
  /// The type returned is the type corresponding to the backend given.
  /// The value returned is +inf if (1) point+dir is outside & moving AWAY from any plane,
  ///     OR (2) when point+dir crosses out a plane BEFORE crossing in ALL other planes.
  /// Note: smin0 parameter is needed here, otherwise smax can become smaller than smin0,
  ///   which means condition (2) happens and +inf must be returned.  Without smin0, this
  ///   condition is sometimes missed.
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToIn(Vector3D<typename Backend::precision_v> const& point,
                                             typename Backend::precision_v const& smin0,
                                             Vector3D<typename Backend::precision_v> const &dir) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t distIn(kInfinity);  // set for earlier returns
    Float_t smax(kInfinity);
    Float_t smin(smin0);

    // hope for a vectorization of this part for Backend==scalar!!
    Bool_t done(Backend::kFalse);
    Float_t pdist[N];
    Float_t proj[N];
    Float_t vdist[N];
    // vectorizable part
    for(int i=0; i<N; ++i) {
      pdist[i] = this->fA[i]*point.x() + this->fB[i]*point.y() + this->fC[i]*point.z() + this->fD[i];
      proj[i] = this->fA[i]*dir.x() + this->fB[i]*dir.y() + this->fC[i]*dir.z();

      // note(SW): on my machine it was better to keep vdist[N] instead of a local variable vdist below
      vdist[i]= -pdist[i]/proj[i];
    }

    // analysis loop
    for(int i=0; i<N; ++i) {
      Bool_t posPoint = pdist[i] >= -kHalfTolerance;
      Bool_t posDir = proj[i] > 0;

      done |= (posPoint && posDir);

      // check if trajectory will intercept plane within current range (smin,smax)

      Bool_t interceptFromInside  = (!posPoint && posDir);
      done |= ( interceptFromInside  && vdist[i]<smin );

      Bool_t interceptFromOutside = (posPoint && !posDir);
      done |= ( interceptFromOutside && vdist[i]>smax );
      if ( Backend::early_returns && IsFull(done) ) return distIn;

      // update smin,smax
      Bool_t validVdist = (smin<vdist[i] && vdist[i]<smax);
      MaskedAssign( interceptFromOutside && validVdist, vdist[i], &smin );
      MaskedAssign( interceptFromInside  && validVdist, vdist[i], &smax );
    }

    // check that entry point found is valid -- cannot be outside (may happen when track is parallel to a face)
    Vector3D<typename Backend::precision_v> entry = point + dir*smin;
    Bool_t valid = Backend::kTrue;
    for(unsigned int i=0; i<N; ++i) {
      Float_t dist = this->fA[i]*entry.x() + this->fB[i]*entry.y() + this->fC[i]*entry.z() + this->fD[i];

      // valid here means if it is not outside plane, or pdist[i]<=0.
      valid &= (dist <= MakePlusTolerant<true>(0.));

    }

    // Return smin, which is the maximum distance in an interceptFromOutside situation
    MaskedAssign( !done && valid, smin, &distIn );

    return distIn;
  }

  /// \return the distance to the planar shell when the point is located within the shell itself.
  /// The type returned is the type corresponding to the backend given.
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t distOut(kInfinity);

    // hope for a vectorization of this part for Backend==scalar !!
    // the idea is to put vectorizable things into this loop
    // and separate the analysis into a separate loop if need be
//  Bool_t done(Backend::kFalse);
    Float_t proj[N];
    Float_t vdist[N];
    for(int i=0; i<N; ++i) {
      Float_t pdist = this->fA[i]*point.x() + this->fB[i]*point.y() + this->fC[i]*point.z() + this->fD[i];

      // early return if point is outside of plane
      //done |= (pdist>0.);
      //if ( IsFull(done) ) return kInfinity;

      proj[i] = this->fA[i]*dir.x() + this->fB[i]*dir.y() + this->fC[i]*dir.z();
      vdist[i] = -pdist / proj[i];
    }

    // add = in vdist[i]>=0  and "proj[i]>0" in order to pass unit tests, but this will slow down DistToOut()!!! check effect!
    for(int i=0; i<N; ++i )
    {
      Bool_t test = ( vdist[i] >= 0. && proj[i]>0 && vdist[i] < distOut );
      MaskedAssign( test, vdist[i], &distOut);
    }

    return distOut;
  }

  /// \return the safety distance to the planar shell when the point is located within the shell itself.
  /// The type returned is the type corresponding to the backend given.
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SafetyToIn(Vector3D<typename Backend::precision_v> const& point,
                  typename Backend::precision_v &safety ) const {

    typedef typename Backend::precision_v Float_t;

    // vectorizable loop
    Float_t dist[N];
    for(int i=0; i<N; ++i) {
      dist[i] = this->fA[i]*point.x() + this->fB[i]*point.y() + this->fC[i]*point.z() + this->fD[i];
    }

    // non-vectorizable part
    for(int i=0; i<N; ++i) {
      MaskedAssign( dist[i]>safety, dist[i], &safety );
    }
    // not necessary: negative answer is fine
    //MaskedAssign(safety<0, 0.0, &safety);
  }


  /// \return the distance to the planar shell when the point is located within the shell itself.
  /// The type returned is the type corresponding to the backend given.
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SafetyToOut(Vector3D<typename Backend::precision_v> const& point,
                   typename Backend::precision_v &safety ) const {

    typedef typename Backend::precision_v Float_t;

    // vectorizable loop
    Float_t dist[N];
    for(int i=0; i<N; ++i) {
      dist[i] = -(this->fA[i]*point.x() + this->fB[i]*point.y() + this->fC[i]*point.z() + this->fD[i]);
    }

    // non-vectorizable part
    for(int i=0; i<N; ++i) {
      MaskedAssign( dist[i]<safety, dist[i], &safety );
    }

    return;
  }

};

} // End global namespace

#endif // VECGEOM_BASE_SIDEPLANES_H_
