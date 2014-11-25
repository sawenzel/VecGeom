/// \file Vector.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include "base/Global.h"

#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace VECGEOM_NAMESPACE {

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
      : fData(vec), fSize(sz), fAllocated(false) {}

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
  Vector<Type>* CopyToGpu(Type *const gpu_ptr_arr,
                          Vector<Type> *const gpu_ptr) const;
#endif

private:

  // Not implemented
  Vector(Vector const &other);
  Vector * operator=(Vector const & other);

};

#ifdef VECGEOM_CUDA

template <typename Type> class Vector;
class VPlacedVolume;

void Vector_CopyToGpu(Precision *const arr, const int size,
                      void *const gpu_ptr);

void Vector_CopyToGpu(VPlacedVolume const **const arr, const int size,
                      void *const gpu_ptr);

#ifdef VECGEOM_CUDA_INTERFACE
template <typename Type>
Vector<Type>* Vector<Type>::CopyToGpu(Type *const gpu_ptr_arr,
                                      Vector<Type> *const gpu_ptr) const {
  Vector_CopyToGpu(gpu_ptr_arr, this->size(), gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}
#endif

#endif // VECGEOM_CUDA

} // End global namespace

#endif // VECGEOM_BASE_CONTAINER_H_
