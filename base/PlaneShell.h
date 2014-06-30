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

};

} // End global namespace

#endif // VECGEOM_BASE_SIDEPLANES_H_
