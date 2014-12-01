/// \file Vector.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include "base/Global.h"

#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( template <typename Type> class Vector; )

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename Type>
class Vector {

private:

  Type *fData;
  int fSize, fMemorySize;
  bool fAllocated;

public:

  VECGEOM_CUDA_HEADER_BOTH
     Vector() : 
  fData(new Type[1]), fSize(0), fMemorySize(1), fAllocated(true) {}

  VECGEOM_CUDA_HEADER_BOTH
  Vector(Type *const vec, const int sz)
     : fData(vec), fSize(sz), fMemorySize(sz), fAllocated(false) {}

  VECGEOM_CUDA_HEADER_BOTH
  Vector(Type *const vec, const int sz, const int maxsize)
     : fData(vec), fSize(sz), fMemorySize(maxsize), fAllocated(true) {}

  ~Vector() {
    if (fAllocated) delete[] fData;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index) {
    return fData[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return fData[index];
  }

  void push_back(const Type item) {
    if (fSize == fMemorySize) {
      assert(fAllocated && "Trying to push on a 'fixed' size vector (memory not allocated by Vector itself");
      fMemorySize = fMemorySize<<1;
      Type *fDataNew = new Type[fMemorySize];
      for (int i = 0; i < fSize; ++i) fDataNew[i] = fData[i];
      delete[] fData;
      fData = fDataNew;
    }
    fData[fSize] = item;
    fSize++;
  }

  typedef Type* iterator;
  typedef Type const* const_iterator;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  iterator begin() const { return &fData[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  iterator end() const { return &fData[fSize]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const_iterator cbegin() const { return &fData[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const_iterator cend() const { return &fData[fSize]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const {
    return fSize;
  }

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::Vector<CudaType_t<Type> > > CopyToGpu(
     DevicePtr<CudaType_t<Type> > const gpu_ptr_arr,
     DevicePtr<cuda::Vector<CudaType_t<Type> > > const gpu_ptr) const 
  {
     gpu_ptr.Construct(gpu_ptr_arr, size());
     return gpu_ptr;
  }

#endif

private:

  // Not implemented
  Vector(Vector const &other);
  Vector * operator=(Vector const & other);

};

} } // End global namespace

#endif // VECGEOM_BASE_CONTAINER_H_
