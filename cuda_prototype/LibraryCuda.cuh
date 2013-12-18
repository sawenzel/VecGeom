#ifndef LIBRARYCUDA_H
#define LIBRARYCUDA_H

#include "mm_malloc.h"
#include "LibraryGeneric.h"

template <>
struct CtTraits<kCuda> {
  typedef float float_t;
  typedef int   int_v;
  typedef float float_v;
  typedef bool  bool_v;
};

typedef CtTraits<kCuda>::int_v   CudaInt;
typedef CtTraits<kCuda>::float_v CudaFloat;
typedef CtTraits<kCuda>::bool_v  CudaBool;
typedef CtTraits<kCuda>::float_t CudaScalar;

template <typename Type>
struct SOA3D<kCuda, Type> {

private:

  int size_;
  Type *a, *b, *c;

public:

  #ifdef STD_CXX11

  SOA3D() : a(nullptr), b(nullptr), c(nullptr), size_(0) {}

  SOA3D(const int size__) : size_(size__) {
    a = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    b = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    c = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
  }

  SOA3D(SOA3D const &other) : SOA3D(other.size_) {
    const int count = other.size_;
    for (int i = 0; i < count; ++i) {
      a[i] = other.a[i];
      b[i] = other.b[i];
      c[i] = other.c[i];
    }
    size_ = count;
  }

  #else

  SOA3D() {
    a = NULL;
    b = NULL;
    c = NULL;
    size_ = 0;
  }

  SOA3D(const int size__) {
    size_ = size__;
    a = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    b = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    c = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
  }

  #endif /* STD_CXX11 */

  #ifndef NVCC

  ~SOA3D() {
    Deallocate();
  }

  #endif /* NVCC */

  SOA3D(Type *const a_, Type *const b_, Type *const c_, const int size__) {
    a = a_;
    b = b_;
    c = c_;
    size_ = size__;
  }

  inline __attribute__((always_inline))
  void Deallocate() {
    if (a) _mm_free(a);
    if (b) _mm_free(b);
    if (c) _mm_free(c);
  }

  #ifdef NVCC

  inline __attribute__((always_inline))
  SOA3D<kCuda, Type> CopyToGPU() const {
    const int count = size();
    const int memsize = count*sizeof(Type);
    Type *a_, *b_, *c_;
    cudaMalloc((void**)&a_, memsize);
    cudaMalloc((void**)&b_, memsize);
    cudaMalloc((void**)&c_, memsize);
    cudaMemcpy(a_, a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_, a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(c_, a, memsize, cudaMemcpyHostToDevice);
    return SOA3D<kCuda, Type>(a_, b_, c_, count);
  }

  inline __attribute__((always_inline))
  void FreeFromGPU() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  #endif /* NVCC */

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  int size() const { return size_; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(a[index], b[index], c[index]);
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type* Memory(const int index) {
    if (index == 0) return a;
    if (index == 1) return b;
    if (index == 2) return c;
    #ifndef NVCC
    throw new std::out_of_range("");
    #endif /* NVCC */
  }

};

typedef SOA3D<kCuda, CtTraits<kCuda>::float_v> SOA3D_CUDA_Float;

__device__
inline __attribute__((always_inline))
int ThreadIndex() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

#endif /* LIBRARYCUDA_H */