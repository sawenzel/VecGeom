#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "base/container.h"
#ifdef VECGEOM_NVCC
#include "backend/cuda_backend.h"
#endif

namespace vecgeom {

template <int arr_size, typename Type>
class Array : public Container<Type> {

private:

  Type arr[size];

public:

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
  int size() const {
    return arr_size;
  }

  #ifdef VECGEOM_NVCC

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  void CopyToGPU(Type *const target) const {
    vecgeom::CopyToGPU(arr, &target, arr_size);
  }

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  void CopyFromGPU(Type *const target) const {
    vecgeom::CopyFromGPU(arr, &target, arr_size);
  }

  #endif

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_ARRAY_H_