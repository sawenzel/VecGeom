/// \file Vector.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include "base/Global.h"

#include "base/Iterator.h"

namespace VECGEOM_NAMESPACE {

template <typename Type>
class Vector {

private:

  Type *fData;
  int fSize, fMemorySize;
  bool fAllocated;

public:

  VECGEOM_CUDA_HEADER_BOTH
  Vector() : fSize(0), fMemorySize(1), fAllocated(true) {
    fData = new Type[fMemorySize];
  }

  VECGEOM_CUDA_HEADER_BOTH
  Vector(Type *const vec, const int size)
      : fData(vec), fSize(size), fAllocated(false) {}

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

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> begin() const {
    return Iterator<Type>(&fData[0]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> end() const {
    return Iterator<Type>(&fData[fSize]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const {
    return fSize;
  }

};

} // End global namespace

#endif // VECGEOM_BASE_CONTAINER_H_
