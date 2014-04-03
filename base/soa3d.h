/**
 * @file soa3d.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "base/global.h"

#include "base/track_container.h"
#include "base/vector3d.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/interface.h"
#endif

namespace vecgeom_cuda { template <typename Type> class SOA3D; }

namespace VECGEOM_NAMESPACE {

/**
 * @brief Stores arrays of three coordinates in a structure of arrays fashion.
 */
template <typename Type>
class SOA3D : public TrackContainer<Type> {

private:

  Type *x_, *y_, *z_;

public:

  /**
   * Initializes the SOA with existing data arrays, performing no allocation.
   */
  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(Type *const x, Type *const y, Type *const z, const unsigned size);

  SOA3D();

  /**
   * Initializes the SOA with a fixed size, allocating an aligned array for each
   * coordinate of the specified size.
   */
  SOA3D(const unsigned size);

  /**
   * Performs a deep copy from another SOA.
   */
  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(TrackContainer<Type> const &other);

  ~SOA3D();

  /**
   * Constructs a vector across all three coordinates from the given index. 
   */ 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(x_[index], y_[index], z_[index]);
  }

  // Element access methods.
  // Can be used to manipulate content if necessary.

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& x(const int index) { return x_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& x(const int index) const { return x_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& y(const int index) { return y_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& y(const int index) const { return y_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& z(const int index) { return z_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& z(const int index) const { return z_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, const Type x, const Type y, const Type z);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, Vector3D<Type> const &vec);

  #ifdef VECGEOM_CUDA_INTERFACE
  /**
   * Allocates and copies the data of this SOA to the GPU, then creates and
   * returns a new SOA object that points to GPU memory.
   */
  SOA3D<Type>* CopyToGpu(Type *const x_gpu, Type *const y_gpu,
                         Type *const z_gpu) const;
  SOA3D<Type>* CopyToGpu(Type *const x_gpu, Type *const y_gpu,
                         Type *const z_gpu, const unsigned size) const;
  #endif // VECGEOM_CUDA_INTERFACE

};

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<Type>::SOA3D(Type *const x, Type *const y, Type *const z,
                   const unsigned size)
    : TrackContainer<Type>(size, true), x_(x), y_(y), z_(z) {}

template <typename Type>
SOA3D<Type>::SOA3D() {
  SOA3D(NULL, NULL, NULL, 0);
}

template <typename Type>
SOA3D<Type>::SOA3D(const unsigned size) : TrackContainer<Type>(size, true) {
  x_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
  y_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
  z_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<Type>::SOA3D(TrackContainer<Type> const &other) {
  SOA3D(other.size());
  const unsigned count = other.size();
  for (int i = 0; i < count; ++i) Set(i, other[i]);
}

template <typename Type>
SOA3D<Type>::~SOA3D() {
  if (this->allocated_) {
    _mm_free(x_);
    _mm_free(y_);
    _mm_free(z_);
  }
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SOA3D<Type>::Set(const int index, const Type x, const Type y,
                      const Type z) {
  x_[index] = x;
  y_[index] = y;
  z_[index] = z;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SOA3D<Type>::Set(const int index, Vector3D<Type> const &vec) {
  x_[index] = vec[0];
  y_[index] = vec[1];
  z_[index] = vec[2];
}

} // End global namespace

namespace vecgeom {

template <typename Type> class SOA3D;

SOA3D<Precision>* SOA3DCopyToGpuInterface(Precision *const x,
                                          Precision *const y,
                                          Precision *const z,
                                          const unsigned size);

#ifdef VECGEOM_CUDA_INTERFACE

template <typename Type>
SOA3D<Type>* SOA3D<Type>::CopyToGpu(Type *const x_gpu,
                                    Type *const y_gpu,
                                    Type *const z_gpu) const {
  const int count = this->size();
  const int mem_size = count*sizeof(Type);
  vecgeom::CopyToGpu(x_, x_gpu, mem_size);
  vecgeom::CopyToGpu(y_, y_gpu, mem_size);
  vecgeom::CopyToGpu(z_, z_gpu, mem_size);
  return SOA3DCopyToGpuInterface(x_gpu, y_gpu, z_gpu, count);
}

template <typename Type>
SOA3D<Type>* SOA3D<Type>::CopyToGpu(Type *const x_gpu,
                                    Type *const y_gpu,
                                    Type *const z_gpu,
                                    const unsigned count) const {
  const unsigned mem_size = count*sizeof(Type);
  vecgeom::CopyToGpu(x_, x_gpu, mem_size);
  vecgeom::CopyToGpu(y_, y_gpu, mem_size);
  vecgeom::CopyToGpu(z_, z_gpu, mem_size);
  return SOA3DCopyToGpuInterface(x_gpu, y_gpu, z_gpu, count);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End namespace vecgeom

#endif // VECGEOM_BASE_SOA3D_H_