#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "base/global.h"

namespace vecgeom {

template <typename Type>
class SOA3D {

private:

  unsigned size_;
  bool allocated_;
  Type *x_, *y_, *z_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(Type *const x, Type *const y, Type *const z, const unsigned size)
      :  size_(size), allocated_(false), x_(x), y_(y), z_(z) {}

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D() {
    SOA3D(NULL, NULL, NULL, 0);
  }

  SOA3D(const unsigned size) : size_(size), allocated_(true) {
    x_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size_, kAlignmentBoundary));
    y_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size_, kAlignmentBoundary));
    z_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size_, kAlignmentBoundary));
  }

  ~SOA3D() {
    Deallocate();
  }

  void Deallocate() {
    if (allocated_) {
      _mm_free(x_);
      _mm_free(y_);
      _mm_free(z_);
    }
  }

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(SOA3D const &other) {
    SOA3D(other.size);
    const unsigned count = other.size;
    for (int i = 0; i < count; ++i) {
      x_[i] = other.z_[i];
      y_[i] = other.y_[i];
      z_[i] = other.z_[i];
    }
    size = count;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  unsigned size() const { return size_; }

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
  void Set(const int index, const Type x, const Type y, const Type z) {
    x_[index] = x;
    y_[index] = y;
    z_[index] = z;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, Vector3D<Type> const &vec) {
    x_[index] = vec[0];
    y_[index] = vec[1];
    z_[index] = vec[2];
  }


  #ifdef VECGEOM_CUDA

  /**
   * Allocates and copies the data of this SOA to the GPU, then creates and
   * returns a new SOA object that points to GPU memory.
   */
  VECGEOM_CUDA_HEADER_HOST
  SOA3D<Type> CopyToGpu() const {
    const int count = size;
    const int mem_size = count*sizeof(Type);
    Type *x, *y, *z;
    cudaMalloc(static_cast<void**>(&x), mem_size);
    cudaMalloc(static_cast<void**>(&y), mem_size);
    cudaMalloc(static_cast<void**>(&z), mem_size);
    cudaMemcpy(x, x_, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(z, z_, mem_size, cudaMemcpyHostToDevice);
    return SOA3D<Type>(x, y, z, count);
  }

  /**
   * Only works for SOA pointing to the GPU.
   */
  VECGEOM_CUDA_HEADER_HOST
  void FreeFromGpu() {
    cudaFree(x_);
    cudaFree(y_);
    cudaFree(z_);
    CudaAssertError();
  }

  #endif // VECGEOM_CUDA

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_SOA3D_H_