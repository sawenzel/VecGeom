/// \file PlaneShell.h
/// \author Guilherme Lima (lima at fnal dot gov)

#ifndef VECGEOM_BASE_SIDEPLANES_H_
#define VECGEOM_BASE_SIDEPLANES_H_

#include "base/Global.h"
//#include "base/Vector3D.h"
// #ifdef VECGEOM_CUDA_INTERFACE
//   #include "backend/cuda/Interface.h"
// #endif

//namespace vecgeom_cuda { template <typename Backend, int N> class PlaneShell; }

#include "backend/Backend.h"
#ifndef VECGEOM_NVCC
  #if (defined(VECGEOM_VC) || defined(VECGEOM_VC_ACCELERATION))
    #include <Vc/Vc>
  #endif
#endif

namespace VECGEOM_NAMESPACE {

/**
 * @brief Uses SoA layout to store arrays of four (plane) parameters,
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
class PlaneShell {

private:

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
    : fA(a), fB(b), fC(c), fD(d)
  { }


  /**
   * Initializes the SOA with a fixed size, allocating an aligned array for each
   * coordinate of the specified size.
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell()
    : fA{0,0,0,0}, fB{0,0,0,0}, fC{0,0,0,0}, fD{0,0,0,0}
  { }

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
    PlaneShell* operator=(PlaneShell const &other) {
      memcpy( this->fA, other.fA, N*sizeof(Type) );
      memcpy( this->fB, other.fB, N*sizeof(Type) );
      memcpy( this->fC, other.fC, N*sizeof(Type) );
      memcpy( this->fD, other.fD, N*sizeof(Type) );
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
    // VcPrecision (point.x(), point.y(), point.z(), 1.0f);
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

  /// \return the distance to the planar shell when the point is located within the shell itself
  /// The type returned is the type corresponding to the backend given
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToOut(Vector3D<typename Backend::precision_v> const& point,
          Vector3D<typename Backend::precision_v> const &dir) const {
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t dist(kInfinity);
    Float_t tmp[N];
    // hope for a vectorization of this part for Backend==scalar !! ( in my case this works )
    for(int i=0; i<N; ++i) {
      Float_t pdist = this->fA[i]*point.x() + this->fB[i]*point.y() + this->fC[i]*point.z() + this->fD[i];
      Float_t proj = this->fA[i]*dir.x() + this->fB[i]*dir.y() + this->fC[i]*dir.z();
      // we could put an early return type of thing here!
      tmp[i] = -pdist / proj;
    }
    for(int i=0; i<N; ++i )
    {
      Bool_t test = ( tmp[i] > 0 && tmp[i] < dist );
      MaskedAssign( test, tmp[i], &dist);
    }
    return dist;
  }

};

} // End global namespace

#endif // VECGEOM_BASE_SIDEPLANES_H_
