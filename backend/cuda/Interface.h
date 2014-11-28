/// \file cuda/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_CUDA_INTERFACE_H_
#define VECGEOM_BACKEND_CUDA_INTERFACE_H_

#include "base/Global.h"

#ifdef VECGEOM_CUDA

#include "driver_types.h" // Required for cudaError_t type
#include "cuda_runtime.h"

namespace vecgeom { namespace cxx {

class VPlacedVolume;
template <typename Type> class SOA3D;
template <typename Type> class AOS3D;

cudaError_t CudaCheckError(const cudaError_t err);

cudaError_t CudaCheckError();

void CudaAssertError(const cudaError_t err);

void CudaAssertError();

cudaError_t CudaMalloc(void** ptr, unsigned size);

cudaError_t CudaCopyToDevice(void* tgt, void const* src, unsigned size);

cudaError_t CudaCopyFromDevice(void* tgt, void const* src, unsigned size);

cudaError_t CudaFree(void* ptr);

template <typename Type>
Type* AllocateOnGpu(const unsigned int size) {
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
void CopyFromGpu(Type const *const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    CudaCopyFromDevice(tgt, src, size)
  );
}

class DevicePtrBase
{
   void *fPtr;

protected:
   DevicePtrBase(const DevicePtrBase&orig) : fPtr(orig.fPtr) {}
   DevicePtrBase &operator=(const DevicePtrBase&orig) { fPtr = orig.fPtr; return *this; }

   void MemcpyToDevice(const void* what, unsigned long nbytes)
   {
      CudaAssertError(cudaMemcpy(fPtr,what,nbytes,
                                 cudaMemcpyHostToDevice) );
   }

   void MemcpyToHostAsync(void* where, unsigned long nbytes, cudaStream_t stream)
   {
      CudaAssertError(cudaMemcpyAsync(where, fPtr, nbytes, cudaMemcpyDeviceToHost, stream));
   }
   void *GetPtr() const { return fPtr; }

   void Malloc(unsigned long size) {
      vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaMalloc((void**)&fPtr, size));
   }
public:
   DevicePtrBase() : fPtr(0) {}
   explicit DevicePtrBase(void *input) : fPtr(input) {}
   ~DevicePtrBase() { /* does not own content per se */ }

};

template <typename Type>
class DevicePtr : private DevicePtrBase
{
public:
   DevicePtr() = default;
   DevicePtr(const DevicePtr&) = default;
   DevicePtr &operator=(const DevicePtr&orig) = default;

   // should be taking a DevicePtr<void*>
   explicit DevicePtr(void *input) : DevicePtrBase(input) {}

   void Allocate(unsigned long nelems = 1) {
      Malloc(nelems*SizeOf());
   }

   template <typename... ArgsTypes>
   void Construct(ArgsTypes... params) const;

   void ToDevice(const Type* what, unsigned long nelems = 1) {
      MemcpyToDevice(what,nelems*SizeOf());
   }
   void FromDevice(Type* where,cudaStream_t stream) {
      // Async since we past a stream.
      MemcpyToHostAsync(where,SizeOf(),stream);
   }
   void FromDevice(Type* where, unsigned long nelems , cudaStream_t stream) {
      // Async since we past a stream.
      MemcpyToHostAsync(where,nelems*SizeOf(),stream);
   }
   operator Type*() const { return reinterpret_cast<Type*>(GetPtr()); }

   static size_t SizeOf();
};

} // End cxx namespace

#ifdef VECGEOM_NVCC

inline namespace cuda {

using vecgeom::cxx::CudaAssertError;
using vecgeom::cxx::CudaMalloc;

template <typename Type>
Type* AllocateOnDevice() {
  Type *ptr;
  vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaMalloc((void**)&ptr, sizeof(Type)));
  return ptr;
}

template <typename DataClass, typename... ArgsTypes>
__global__
void ConstructOnGpu(DataClass *const gpu_ptr, ArgsTypes... params) {
   new (gpu_ptr) DataClass(params...);
}

template <typename DataClass, typename... ArgsTypes>
void Generic_CopyToGpu(DataClass *const gpu_ptr, ArgsTypes... params)
{
   ConstructOnGpu<<<1, 1>>>(gpu_ptr, params...);
}

template <typename DataClass, typename... ArgsTypes>
void DevicePtr::Construct(ArgsTypes... params) const
{
   ConstructOnGpu<<<1, 1>>>(*(*this), params...);
}

template <typename DataClass> size_t DevicePtr::SizeOf()
{
   return sizeof<DataClass>;
}

} // End cuda namespace

#else // VECGEOM_NVCC

namespace cuda {

template <typename Type> Type* AllocateOnDevice();
template <typename DataClass, typename... ArgsTypes>
void Generic_CopyToGpu(DataClass *const gpu_ptr, ArgsTypes... params);

} // End cuda namespace

#endif // VECGEOM_NVCC

} // End vecgeom namespace

#endif // VECGEOM_CUDA

#endif // VECGEOM_BACKEND_CUDA_INTERFACE_H_
