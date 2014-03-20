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

template <typename Type>
Type* AllocateOnGpu(const unsigned size);

template <typename Type>
Type* AllocateOnGpu();

template <typename Type>
void FreeFromGpu(Type *const ptr);

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt, const unsigned size);

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt);

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt, const unsigned size);

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt);

void CudaCheckMemory(size_t *const free_memory, size_t *const total_memory);

} // End global namespace

#endif // VECGEOM_BACKEND_CUDA_INTERFACE_H_