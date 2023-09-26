/// \file cuda/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_CUDA_INTERFACE_H_
#define VECGEOM_BACKEND_CUDA_INTERFACE_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"

#ifdef VECGEOM_ENABLE_CUDA

#include "driver_types.h" // Required for cudaError_t type
#include "cuda_runtime.h"

#include <vector>
#include <unordered_map>
#include <type_traits>

namespace vecgeom {

#ifdef VECCORE_CUDA

inline namespace cuda {

template <typename DataClass, typename... ArgsTypes>
__global__ void ConstructOnGpu(DataClass *gpu_ptr, ArgsTypes... params)
{
  new (gpu_ptr) DataClass(params...);
}

template <typename DataClass, typename... ArgsTypes>
__global__ void ConstructArrayOnGpu(DataClass *gpu_ptr, size_t nElements, ArgsTypes... params)
{

  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int idx = tid;
  while (idx < nElements) {
    new (gpu_ptr + idx) DataClass(params...);
    idx += blockDim.x * gridDim.x;
  }
}

/*!
 * Construct many objects on the GPU, whose addresses and parameters are passed as arrays.
 * \tparam DataClass Type of the objects to construct.
 * \param nElements Number of elements to construct. It is assumed that all argument arrays have this size.
 * \param gpu_ptrs  Array of pointers to place the new objects at.
 * \param params    Array(s) of constructor parameters for each object.
 */
template <typename DataClass, typename... ArgsTypes>
__global__ void ConstructManyOnGpu_kernel(size_t nElements, DataClass **gpu_ptrs, const ArgsTypes *... params)
{
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (size_t idx = tid; idx < nElements; idx += blockDim.x * gridDim.x) {
    new (gpu_ptrs[idx]) DataClass(params[idx]...);
  }
}

template <typename DataClass>
__global__ void CopyBBoxesToGpu(size_t nElements, DataClass **raw_ptrs, Precision *boxes)
{
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (size_t idx = tid; idx < nElements; idx += blockDim.x * gridDim.x) {
    raw_ptrs[idx]->SetBBox({boxes[6 * idx], boxes[6 * idx + 1], boxes[6 * idx + 2]},
                           {boxes[6 * idx + 3], boxes[6 * idx + 4], boxes[6 * idx + 5]});
  }
}

template <typename DataClass, typename... ArgsTypes>
void Generic_CopyToGpu(DataClass *const gpu_ptr, ArgsTypes... params)
{
  ConstructOnGpu<<<1, 1>>>(gpu_ptr, params...);
}

} // namespace cuda

#else

namespace cuda {

template <typename Type>
Type *AllocateOnDevice();
template <typename DataClass, typename... ArgsTypes>
void Generic_CopyToGpu(DataClass *const gpu_ptr, ArgsTypes... params);

} // namespace cuda

#endif

#ifdef VECCORE_CUDA
namespace cxx {
#else
inline namespace cxx {
#endif

cudaError_t CudaCheckError(const cudaError_t err);

cudaError_t CudaCheckError();

void CudaAssertError(const cudaError_t err);

void CudaAssertError();

cudaError_t CudaMalloc(void **ptr, unsigned size);

cudaError_t CudaCopyToDevice(void *tgt, void const *src, unsigned size);

cudaError_t CudaCopyFromDevice(void *tgt, void const *src, unsigned size);

cudaError_t CudaCopyFromDeviceAsync(void *tgt, void const * src, unsigned size, cudaStream_t stream);

cudaError_t CudaFree(void *ptr);

cudaError_t CudaDeviceSetStackLimit(unsigned size);

cudaError_t CudaDeviceSetHeapLimit(unsigned size);

template <typename Type>
Type *AllocateOnDevice()
{
  Type *ptr;
  vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaMalloc((void **)&ptr, sizeof(Type)));
  return ptr;
}

template <typename Type>
Type *AllocateOnGpu(const unsigned int size)
{
  Type *ptr;
  vecgeom::cxx::CudaAssertError(CudaMalloc((void **)&ptr, size));
  return ptr;
}

template <typename Type>
Type *AllocateOnGpu()
{
  return AllocateOnGpu<Type>(sizeof(Type));
}

template <typename Type>
void FreeFromGpu(Type *const ptr)
{
  vecgeom::cxx::CudaAssertError(CudaFree(ptr));
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt, const unsigned size)
{
  vecgeom::cxx::CudaAssertError(CudaCopyToDevice(tgt, src, size));
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt)
{
  CopyToGpu<Type>(src, tgt, sizeof(Type));
}

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt, const unsigned size)
{
  vecgeom::cxx::CudaAssertError(CudaCopyFromDevice(tgt, src, size));
}

class DevicePtrBase {
  void *fPtr;
#ifdef DEBUG_DEVICEPTR
  size_t fAllocatedSize;
  bool fIncremented;
#endif

protected:
  DevicePtrBase(const DevicePtrBase &orig)
      : fPtr(orig.fPtr)
#ifdef DEBUG_DEVICEPTR
        ,
        fAllocatedSize(0), fIncremented(false)
#endif
  {
  }

  DevicePtrBase &operator=(const DevicePtrBase &orig)
  {
    fPtr = orig.fPtr;
#ifdef DEBUG_DEVICEPTR
    fAllocatedSize = orig.fAllocatedSize;
    fIncremented   = orig.fIncremented;
#endif
    return *this;
  }

  void MemcpyToDevice(const void *what, unsigned long nbytes)
  {
    if (nbytes) vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaCopyToDevice(fPtr, what, nbytes));
  }

  void MemcpyToHostAsync(void *where, unsigned long nbytes, cudaStream_t stream)
  {
    vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaCopyFromDeviceAsync(where, fPtr, nbytes, stream));
  }

  VECCORE_ATT_HOST_DEVICE
  void *GetPtr() const { return fPtr; }

  void Free()
  {
    vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaFree((void *)fPtr));
#ifdef DEBUG_DEVICEPTR
    fAllocatedSize = 0;
#endif
  }

  void Increment(long add)
  {
    fPtr = (char *)fPtr + add;
#ifdef DEBUG_DEVICEPTR
    if (add) fIncremented = true;
#endif
  }

public:
  DevicePtrBase()
      : fPtr(0)
#ifdef DEBUG_DEVICEPTR
        ,
        fAllocatedSize(0), fIncremented(0)
#endif
  {
  }

  explicit DevicePtrBase(void *input)
      : fPtr(input)
#ifdef DEBUG_DEVICEPTR
        ,
        fAllocatedSize(0), fIncremented(0)
#endif
  {
  }

  ~DevicePtrBase()
  { /* does not own content per se */
  }

  void Malloc(unsigned long size)
  {
    vecgeom::cxx::CudaAssertError(vecgeom::cxx::CudaMalloc((void **)&fPtr, size));
#ifdef DEBUG_DEVICEPTR
    fAllocatedSize = size;
#endif
  }
};

template <typename T>
class DevicePtr;

template <typename Type, typename Derived = DevicePtr<Type>>
class DevicePtrImpl : public DevicePtrBase {
protected:
  DevicePtrImpl(const DevicePtrImpl & /* orig */) = default;
  DevicePtrImpl &operator=(const DevicePtrImpl & /*orig*/) = default;
  DevicePtrImpl()                                          = default;
  explicit DevicePtrImpl(void *input) : DevicePtrBase(input) {}
  ~DevicePtrImpl() = default;

public:
  void Allocate(unsigned long nelems = 1) { Malloc(nelems * Derived::SizeOf()); }

  void Deallocate() { Free(); }

  void ToDevice(const Type *what, unsigned long nelems = 1) { MemcpyToDevice(what, nelems * Derived::SizeOf()); }
  void FromDevice(Type *where, cudaStream_t stream)
  {
    // Async since we pass a stream.
    MemcpyToHostAsync(where, Derived::SizeOf(), stream);
  }
  void FromDevice(Type *where, unsigned long nelems, cudaStream_t stream)
  {
    // Async since we pass a stream.
    MemcpyToHostAsync(where, nelems * Derived::SizeOf(), stream);
  }

  VECCORE_ATT_HOST_DEVICE
  Type *GetPtr() const { return reinterpret_cast<Type *>(DevicePtrBase::GetPtr()); }

  VECCORE_ATT_HOST_DEVICE
  operator Type *() const { return GetPtr(); }

  Derived &operator++() // prefix ++
  {
    Increment(Derived::SizeOf());
    return *(Derived *)this;
  }

  Derived operator++(int) // postfix ++
  {
    Derived tmp(*(Derived *)this);
    Increment(Derived::SizeOf());
    return tmp;
  }

  Derived operator+(const size_t& rhs)
  {
    Derived tmp(*(Derived *)this);
    tmp.Increment(rhs);
    return tmp;
  }

  Derived &operator+=(long len) // prefix ++
  {
    Increment(len * Derived::SizeOf());
    return *(Derived *)this;
  }
};

template <typename Type>
class DevicePtr : public DevicePtrImpl<Type> {
public:
  DevicePtr()                  = default;
  DevicePtr(const DevicePtr &) = default;
  DevicePtr &operator=(const DevicePtr &orig) = default;

  // should be taking a DevicePtr<void*>
  explicit DevicePtr(void *input) : DevicePtrImpl<Type>(input) {}

  // Need to go via the explicit route accepting all conversion
  // because the regular c++ compilation
  // does not actually see the declaration for the cuda version
  // (and thus can not determine the inheritance).
  template <typename inputType>
  explicit DevicePtr(DevicePtr<inputType> const &input) : DevicePtrImpl<Type>((void *)input)
  {
  }

  // Disallow conversion from const to non-const.
  DevicePtr(DevicePtr<const Type> const &input,
            typename std::enable_if<!std::is_const<Type>::value, Type>::type * = nullptr) = delete;

#ifdef VECCORE_CUDA
  // Allows implicit conversion from DevicePtr<Derived> to DevicePtr<Base>
  template <typename inputType, typename std::enable_if<std::is_base_of<Type, inputType>::value>::type * = nullptr>
  DevicePtr(DevicePtr<inputType> const &input) : DevicePtrImpl<Type>(input.GetPtr())
  {
  }

  // Disallow conversion from const to non-const.
  template <typename inputType, typename std::enable_if<std::is_base_of<Type, inputType>::value>::type * = nullptr>
  DevicePtr(DevicePtr<const inputType> const &input) = delete;
#endif

#ifdef VECCORE_CUDA
  template <typename... ArgsTypes>
  void Construct(ArgsTypes... params) const
  {
    ConstructOnGpu<<<1, 1>>>(this->GetPtr(), params...);
  }

  template <typename... ArgsTypes>
  void ConstructArray(size_t nElements, ArgsTypes... params) const
  {
    ConstructArrayOnGpu<<<nElements, 1>>>(this->GetPtr(), nElements, params...);
  }

  static size_t SizeOf() { return sizeof(Type); }

#else
  template <typename... ArgsTypes>
  void Construct(ArgsTypes... params) const;
  template <typename... ArgsTypes>
  void ConstructArray(size_t nElements, ArgsTypes... params) const;

  static size_t SizeOf();
#endif
};

template <typename Type>
class DevicePtr<const Type> : private DevicePtrImpl<const Type> {
public:
  DevicePtr()                  = default;
  DevicePtr(const DevicePtr &) = default;
  DevicePtr &operator=(const DevicePtr &orig) = default;

  // should be taking a DevicePtr<void*>
  explicit DevicePtr(void *input) : DevicePtrBase(input) {}

  // Need to go via the explicit route accepting all conversion
  // because the regular c++ compilation
  // does not actually see the declaration for the cuda version
  // (and thus can not determine the inheritance).
  template <typename inputType>
  explicit DevicePtr(DevicePtr<inputType> const &input) : DevicePtrImpl<const Type>((void *)input)
  {
  }

  // Implicit conversion from non-const to const.
  DevicePtr(DevicePtr<typename std::remove_const<Type>::type> const &input) : DevicePtrImpl<const Type>((void *)input)
  {
  }

#ifdef VECCORE_CUDA
  // Allows implicit conversion from DevicePtr<Derived> to DevicePtr<Base>
  template <typename inputType, typename std::enable_if<std::is_base_of<Type, inputType>::value>::type * = nullptr>
  DevicePtr(DevicePtr<inputType> const &input) : DevicePtrImpl<const Type>(input.GetPtr())
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  const Type *GetPtr() const { return reinterpret_cast<const Type *>(DevicePtrBase::GetPtr()); }

  VECCORE_ATT_HOST_DEVICE
  operator const Type *() const { return GetPtr(); }

#ifdef VECCORE_CUDA
  template <typename DataClass, typename... ArgsTypes>
  void Construct(ArgsTypes... params) const
  {
    ConstructOnGpu<<<1, 1>>>(*(*this), params...);
  }

  template <typename... ArgsTypes>
  void ConstructArray(size_t nElements, ArgsTypes... params) const
  {
    ConstructArrayOnGpu<<<nElements, 1>>>(this->GetPtr(), nElements, params...);
  }

  static size_t SizeOf() { return sizeof(Type); }

#else
  template <typename... ArgsTypes>
  void Construct(ArgsTypes... params) const;
  template <typename... ArgsTypes>
  void ConstructArray(size_t nElements, ArgsTypes... params) const;

  static size_t SizeOf();
#endif
};

namespace CudaInterfaceHelpers {

/*!
 * Copy multiple arrays of values to the GPU.
 * For each array, allocate memory on the device, and copy it to the GPU.
 * The cpuToGpuMapping finally maps the CPU array pointers to the GPU arrays.
 * \param[out] cpuToGpuMapping Mapping of CPU array to GPU array. It gets filled during the function execution.
 * \param[in]  nElement Number of elements in all collections.
 * \param[in]  toCopy First array to copy.
 * \param[in]  restToCopy Parameter pack with more arrays to copy (can be empty).
 */
template <typename Arg_t, typename... Args_t>
void allocateAndCopyToGpu(std::unordered_map<const void *, void *> &cpuToGpuMapping, std::size_t nElement,
                          const Arg_t *toCopy, const Args_t *... restToCopy)
{
  const auto nByte         = sizeof(toCopy[0]) * nElement;
  const void *hostMem      = toCopy;
  void *deviceMem          = AllocateOnGpu<void *>(nByte);
  cpuToGpuMapping[hostMem] = deviceMem;
  CopyToGpu(hostMem, deviceMem, nByte);

#if __cplusplus >= 201703L
  if constexpr (sizeof...(Args_t) > 0) {
    allocateAndCopyToGpu(cpuToGpuMapping, nElement, restToCopy...);
  }
#else
  // C++11 "fold expression" hack. Please remove once VecGeom moves to c++17.
  int expandParameterPack[] = {0, ((void)allocateAndCopyToGpu(cpuToGpuMapping, nElement, restToCopy), 0)...};
  (void)expandParameterPack[0]; // Make nvcc happy
#endif
}

} // namespace CudaInterfaceHelpers

/*!
 * Construct many objects on the GPU, whose addresses and constructor parameters are passed as arrays.
 * \tparam DataClass The type to construct on the GPU.
 * \tparam DevPtr_t Device pointer type to specify the location of the GPU objects.
 * \param  nElement Number of elements to construct. It is assumed that all argument arrays have this length.
 * \param  gpu_ptrs Array of addresses to place the new objects at.
 * \param  params   Array(s) of constructor parameters with one entry for each object.
 */
template <class DataClass, class DevPtr_t, typename... Args_t>
void ConstructManyOnGpu(std::size_t nElement, const DevPtr_t *gpu_ptrs, const Args_t *... params)
#ifdef VECCORE_CUDA
{
  using namespace CudaInterfaceHelpers;
  std::unordered_map<const void *, void *> cpuToGpuMem;
  std::vector<DataClass *> raw_gpu_ptrs;
  std::transform(gpu_ptrs, gpu_ptrs + nElement, std::back_inserter(raw_gpu_ptrs),
                 [](const DevPtr_t &ptr) { return static_cast<DataClass *>(ptr.GetPtr()); });
  allocateAndCopyToGpu(cpuToGpuMem, nElement, raw_gpu_ptrs.data(), params...);

  ConstructManyOnGpu_kernel<<<128, 32>>>(raw_gpu_ptrs.size(),
                                  static_cast<decltype(raw_gpu_ptrs.data())>(cpuToGpuMem[raw_gpu_ptrs.data()]),
                                  static_cast<decltype(params)>(cpuToGpuMem[params])...);

  for (const auto &memCpu_memGpu : cpuToGpuMem) {
    FreeFromGpu(memCpu_memGpu.second);
  }

  CudaCheckError();
}
#else
    ;
#endif

template <class DataClass, class DevPtr_t>
void CopyBBoxesToGpuImpl(std::size_t nElement, const DevPtr_t *gpu_ptrs, Precision *boxes_data)
#ifdef VECCORE_CUDA
{
  std::unordered_map<const void *, void *> cpuToGpuMem;
  std::vector<DataClass *> raw_gpu_ptrs;
  std::transform(gpu_ptrs, gpu_ptrs + nElement, std::back_inserter(raw_gpu_ptrs),
                 [](const DevPtr_t &ptr) { return static_cast<DataClass *>(ptr.GetPtr()); });

  const auto nByteBoxes        = 6 * nElement * sizeof(Precision);
  const auto nByteVolumes      = nElement * sizeof(DataClass *);
  Precision *boxes_data_gpu    = AllocateOnGpu<Precision>(nByteBoxes);
  DataClass **raw_gpu_ptrs_gpu = AllocateOnGpu<DataClass *>(nByteVolumes);

  CopyToGpu(boxes_data, boxes_data_gpu, nByteBoxes);
  CopyToGpu(raw_gpu_ptrs.data(), raw_gpu_ptrs_gpu, nByteVolumes);
  cudaDeviceSynchronize();

  CopyBBoxesToGpu<DataClass><<<128, 32>>>(raw_gpu_ptrs.size(), raw_gpu_ptrs_gpu, boxes_data_gpu);

  FreeFromGpu(boxes_data_gpu);
  FreeFromGpu(raw_gpu_ptrs_gpu);
}
#else
    ;
#endif

} // End cxx namespace

} // namespace vecgeom

#endif // VECGEOM_ENABLE_CUDA

#endif // VECGEOM_BACKEND_CUDA_INTERFACE_H_
