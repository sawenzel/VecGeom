/**
 * @file cuda/backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_CUDA_INTERFACE_H_
#define VECGEOM_BACKEND_CUDA_INTERFACE_H_

#include "base/global.h"

namespace vecgeom {

#ifndef VECGEOM_NVCC
typedef int cudaError_t;
#endif

cudaError_t CudaCheckError(const cudaError_t err);

cudaError_t CudaCheckError();

void CudaAssertError(const cudaError_t err);

void CudaAssertError();

cudaError_t CudaMalloc(void** ptr, unsigned size);

cudaError_t CudaCopyToDevice(void* tgt, void const* src, unsigned size);

cudaError_t CudaCopyFromDevice(void* tgt, void const* src, unsigned size);

cudaError_t CudaFree(void* ptr);

template <typename Type>
Type* AllocateOnGpu(const unsigned size) {
  Type *ptr;
  CudaAssertError(CudaMalloc((void**)&ptr, size));
  return ptr;
}

template <typename Type>
Type* AllocateOnGpu() {
  return AllocateOnGpu<Type>(sizeof(Type));
}

template <typename Type>
void FreeFromGpu(Type *const ptr) {
  CudaAssertError(CudaFree(ptr));
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    CudaCopyToDevice(tgt, src, size)
  );
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt) {
  CopyToGpu<Type>(src, tgt, sizeof(Type));
}

template <typename Type>
void CopyFromGpu(Type const * const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    CudaCopyFromDevice(tgt, src, size)
  );
}

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt) {
  CopyFromGpu<Type>(src, tgt, sizeof(Type));
}

} // End global namespace

#endif // VECGEOM_BACKEND_CUDA_INTERFACE_H_