#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "base/container.h"
#ifdef VECGEOM_NVCC
#include "backend/cuda_backend.h"
#endif

namespace vecgeom {

template <typename Type>
class Array : public Container<Type> {

private:

  Type *arr;
  const int size_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  Array(Type const *data, const int size_) {
    size = size_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index) {
    return arr[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return arr[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int Size() const {
    return size;
  }

  #ifdef VECGEOM_NVCC

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  void CopyToGPU(Type *const target) const {
    vecgeom::CopyToGPU(arr, &target, size);
  }

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  void CopyFromGPU(Type *const target) const {
    vecgeom::CopyFromGPU(arr, &target, size);
  }

  #endif

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_ARRAY_H_