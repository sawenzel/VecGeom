#include "base/array.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda_backend.cuh"
#endif

namespace vecgeom {

#ifdef VECGEOM_CUDA

namespace {

template <typename Type>
__global__
void ConstructOnGpu(Type const *const arr, const int size,
                    Array<Type> *const gpu_ptr) {
  new(gpu_ptr) Array<Type>(arr, size);
}

} // End anonymous namespace

template <typename Type>
Array<Type>* Container<Type>::CopyToGpu(Type const *const gpu_ptr_arr,
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
  Type *arr_gpu = vecgeom::AllocateOnGpu<Type>(this->size()*sizeof(Type));
  vecgeom::CopyToGpu(arr, arr_gpu, this->size()*sizeof(Type));
  delete arr;
  return arr_gpu;
}

#endif // VECGEOM_CUDA

} // End namespace vecgeom