/// \file Vector.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include <initializer_list>
#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class Vector;);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, Vector, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Internal {
template <typename T>
struct AllocTrait {

  // Allocate raw buffer to hold the element.
  VECCORE_ATT_HOST_DEVICE
  static T *Allocate(size_t nElems) { return reinterpret_cast<T *>(new char[nElems * sizeof(T)]); }

  // Release raw buffer to hold the element.
  VECCORE_ATT_HOST_DEVICE
  static void Deallocate(T *startBuffer) { delete[]((char *)startBuffer); }

  VECCORE_ATT_HOST_DEVICE
  static void Destroy(T &obj) { obj.~T(); };

  VECCORE_ATT_HOST_DEVICE
  static void Destroy(T *arr, size_t nElem)
  {
    for (size_t i = 0; i < nElem; ++i)
      Destroy(arr[i]);
  }
};

template <typename T>
struct AllocTrait<T *> {

  // Allocate raw buffer to hold the element.
  VECCORE_ATT_HOST_DEVICE
  static T **Allocate(size_t nElems)
  {
    T **ptr = reinterpret_cast<T **>(new char[nElems * sizeof(T *)]);
    assert(ptr != nullptr && "Error: Memory allocation failed! If on GPU, consider increasing the heap size on GPU "
                             "with CudaDeviceSetHeapLimit(new_size)");
    return ptr;
  }

  // Release raw buffer to hold the element.
  VECCORE_ATT_HOST_DEVICE
  static void Deallocate(T **startBuffer) { delete[]((char *)startBuffer); }

  VECCORE_ATT_HOST_DEVICE
  static void Destroy(T *&) {}

  VECCORE_ATT_HOST_DEVICE
  static void Destroy(T ** /*arr*/, size_t /*nElem*/) {}
};
} // namespace Internal

template <typename Type>
class VectorBase {

private:
  Type *fData;
  size_t fSize, fMemorySize;
  bool fAllocated;

public:
  using value_type = Type;

  VECCORE_ATT_HOST_DEVICE
  VectorBase() : VectorBase(5) {}

  VECCORE_ATT_HOST_DEVICE
  VectorBase(size_t maxsize) : fData(nullptr), fSize(0), fMemorySize(0), fAllocated(true) { reserve(maxsize); }

  VECCORE_ATT_HOST_DEVICE
  VectorBase(Type *const vec, const int sz) : fData(vec), fSize(sz), fMemorySize(sz), fAllocated(false) {}

  VECCORE_ATT_HOST_DEVICE
  VectorBase(Type *const vec, const int sz, const int maxsize)
      : fData(vec), fSize(sz), fMemorySize(maxsize), fAllocated(false)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  VectorBase(VectorBase const &other) : fSize(other.fSize), fMemorySize(other.fMemorySize), fAllocated(true)
  {
    fData = Internal::AllocTrait<Type>::Allocate(fMemorySize);
    for (size_t i = 0; i < fSize; ++i)
      new (&fData[i]) Type(other.fData[i]);
  }

  VECCORE_ATT_HOST_DEVICE
  VectorBase &operator=(VectorBase const &other)
  {
    if (&other != this) {
      reserve(other.fMemorySize);
      for (size_t i = 0; i < other.fSize; ++i)
        push_back(other.fData[i]);
    }
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  VectorBase(std::initializer_list<Type> entries)
  {
    fSize       = entries.size();
    fData       = Internal::AllocTrait<Type>::Allocate(fSize);
    fAllocated  = true;
    fMemorySize = entries.size() * sizeof(Type);
    for (auto itm : entries)
      this->push_back(itm);
  }

  VECCORE_ATT_HOST_DEVICE
  ~VectorBase()
  {
    if (fAllocated) Internal::AllocTrait<Type>::Deallocate(fData);
  }

  VECCORE_ATT_HOST_DEVICE
  void clear()
  {
    Internal::AllocTrait<Type>::Destroy(fData, fSize);
    fSize = 0;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &operator[](const int index) { return fData[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &operator[](const int index) const { return fData[index]; }

  VECCORE_ATT_HOST_DEVICE
  void push_back(const Type item)
  {
    if (fSize == fMemorySize) {
      assert(fAllocated && "Trying to push on a 'fixed' size vector (memory "
                           "not allocated by Vector itself)");
      reserve(fMemorySize << 1);
    }
    new (&fData[fSize]) Type(item);
    fSize++;
  }

  typedef Type *iterator;
  typedef Type const *const_iterator;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  iterator begin() const { return &fData[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  iterator end() const { return &fData[fSize]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const_iterator cbegin() const { return &fData[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const_iterator cend() const { return &fData[fSize]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t size() const { return fSize; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t capacity() const { return fMemorySize; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void resize(size_t newsize, Type value)
  {
    if (newsize <= fSize) {
      for (size_t i = newsize; i < fSize; ++i) {
        Internal::AllocTrait<Type>::Destroy(fData[i]);
      }
      fSize = newsize;
    } else {
      if (newsize > fMemorySize) {
        reserve(newsize);
      }
      for (size_t i = fSize; i < newsize; ++i)
        push_back(value);
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void reserve(size_t newsize)
  {
    if (newsize <= fMemorySize) {
      // Do nothing ...
    } else {
      Type *newdata = Internal::AllocTrait<Type>::Allocate(newsize);
      for (size_t i = 0; i < fSize; ++i)
        new (&newdata[i]) Type(fData[i]);
      Internal::AllocTrait<Type>::Destroy(fData, fSize);
      if (fAllocated) {
        Internal::AllocTrait<Type>::Deallocate(fData);
      }
      fData       = newdata;
      fMemorySize = newsize;
      fAllocated  = true;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  iterator erase(const_iterator position)
  {
    iterator where = (begin() + (position - cbegin()));
    if (where + 1 != end()) {
      auto last = cend();
      for (auto c = where; (c + 1) != last; ++c)
        *c = *(c + 1);
    }
    --fSize;
    if (fSize) Internal::AllocTrait<Type>::Destroy(fData[fSize]);
    return where;
  }
};

template <typename Type>
class Vector : public VectorBase<Type> {
public:
  using VectorBase<Type>::VectorBase;
  using typename VectorBase<Type>::iterator;
  using typename VectorBase<Type>::const_iterator;

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::Vector<CudaType_t<Type>>> CopyToGpu(DevicePtr<CudaType_t<Type>> const gpu_ptr_arr,
                                                      DevicePtr<cuda::Vector<CudaType_t<Type>>> const gpu_ptr) const
  {
    gpu_ptr.Construct(gpu_ptr_arr, VectorBase<Type>::size());
    return gpu_ptr;
  }
#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_
