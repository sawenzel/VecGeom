#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "mm_malloc.h"
#include "base/utilities.h"
#include "base/types.h"

template <typename Type>
struct SOA3D {

private:

  int size;
  bool allocated;
  Type *a, *b, *c;

public:

  #ifdef VECGEOM_STD_CXX11

  SOA3D(Type *const a_, Type *const b_, Type *const c_, const int size_)
      : a(a_), b(b_), c(c_), size(size_), allocated(false) {};

  SOA3D(const int size_) : size(size_), allocated(true) {
    a = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
    b = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
    c = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
  }

  SOA3D() : SOA3D(nullptr, nullptr, nullptr, 0) {};

  #else // VECGEOM_STD_CXX11

  SOA3D(Type *const a_, Type *const b_, Type *const c_, const int size_) {
    a = a_;
    b = b_;
    c = c_;
    size = size_;
    allocated = false;
  }

  SOA3D(const int size_) {
    size = size_;
    allocated = true;
    a = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
    b = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
    c = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
  }

  SOA3D() : SOA3D(NULL, NULL, NULL, 0) {}

  #endif /* VECGEOM_STD_CXX11 */

  SOA3D(SOA3D const &other) : SOA3D(other.size) {
    const int count = other.size;
    for (int i = 0; i < count; ++i) {
      a[i] = other.a[i];
      b[i] = other.b[i];
      c[i] = other.c[i];
    }
    size = count;
  }

  ~SOA3D() {
    Deallocate();
  }

  VECGEOM_INLINE
  void Deallocate() {
    if (allocated) {
      _mm_free(a);
      _mm_free(b);
      _mm_free(c);
    }
  }

  #ifdef VECGEOM_NVCC

  /**
   * Allocates and copies the data of this SOA to the GPU, then creates and
   * returns a new SOA object that points to GPU memory.
   */
  VECGEOM_INLINE
  SOA3D<Type> CopyToGPU() const {
    const int count = size();
    const int mem_size = count*sizeof(Type);
    Type *a_, *b_, *c_;
    cudaMalloc(static_cast<void**>(&a_), mem_size);
    cudaMalloc(static_cast<void**>(&b_), mem_size);
    cudaMalloc(static_cast<void**>(&c_), mem_size);
    cudaMemcpy(a_, a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_, b, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_, c, mem_size, cudaMemcpyHostToDevice);
    return SOA3D<Type>(a_, b_, c_, count);
  }

  /**
   * Only works for SOA pointing to the GPU, but will fail silently if this is
   * not the case.
   */
  VECGEOM_INLINE
  void FreeFromGPU() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  #endif // VECGEOM_NVCC

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int Size() const { return size; }

  /**
   * Constructs a vector across all three coordinates from the given index. 
   */ 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(a[index], b[index], c[index]);
  }

  // Element access methods.
  // Can be used to manipulate content if necessary.

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& X(const int index) {
    return a[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& X(const int index) const {
    return a[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& Y(const int index) {
    return b[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& Y(const int index) const {
    return b[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& Z(const int index) {
    return c[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& Z(const int index) const {
    return c[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, const Type a_, const Type b_, const Type c_) {
    a[index] = a_;
    b[index] = b_;
    c[index] = c_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, Vector3D<Type> const &vec) {
    a[index] = vec[0];
    b[index] = vec[1];
    c[index] = vec[2];
  }

};

#endif // VECGEOM_BASE_SOA3D_H_