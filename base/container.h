#ifndef VECGEOM_BASE_CONTAINER_H_
#define VECGEOM_BASE_CONTAINER_H_

#include "base/global.h"
#include "base/array.h"
#include "base/iterator.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda_backend.cuh"
#endif

namespace vecgeom {

template <typename Type>
class Container {

public:

  VECGEOM_CUDA_HEADER_BOTH
  Container() {}

  virtual ~Container() {}

  VECGEOM_CUDA_HEADER_BOTH
  virtual Iterator<Type> begin() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Iterator<Type> end() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual int size() const =0;

  #ifdef VECGEOM_CUDA
  Array<Type>* CopyToGpu(Type *const gpu_ptr_arr,
                         Array<Type> *const gpu_ptr) const;
  Array<Type>* CopyToGpu() const;
  Type* CopyContentToGpu() const;
  #endif

};

#ifdef VECGEOM_CUDA

namespace {

template <typename Type>
__global__
void ConstructOnGpu(Type *const arr, const int size,
                    Array<Type> *const gpu_ptr) {
  new(gpu_ptr) vecgeom::Array<Type>(arr, size);
}

} // End anonymous namespace

template <typename Type>
Array<Type>* Container<Type>::CopyToGpu(Type *const gpu_ptr_arr,
                                        Array<Type> *const gpu_ptr) const {
  ConstructOnGpu<<<1, 1>>>(gpu_ptr_arr, this->size(), gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

template <typename Type>
Array<Type>* Container<Type>::CopyToGpu() const {
  Type *arr_gpu = CopyContentToGpu();
  Array<Type> *const gpu_ptr = AllocateOnGpu<Array<Type> >();
  return CopyToGpu(arr_gpu, gpu_ptr);
}

template <typename Type>
Type* Container<Type>::CopyContentToGpu() const {
  Type *const arr = new Type[this->size()];
  int i = 0;
  for (Iterator<Type> j = this->begin(); j != this->end(); ++j) {
    arr[i] = *j;
    i++;
  }
  Type *const arr_gpu = vecgeom::AllocateOnGpu<Type>(this->size()*sizeof(Type));
  vecgeom::CopyToGpu(arr, arr_gpu, this->size()*sizeof(Type));
  delete arr;
  return arr_gpu;
}

#endif // VECGEOM_CUDA

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_