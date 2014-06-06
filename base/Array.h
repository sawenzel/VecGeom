/// \file Array.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/ConstIterator.h"
#include "base/Iterator.h"

namespace VECGEOM_NAMESPACE {

template <typename Type>
class Array : public AlignedBase {

private:

  Type *fData;
  int fSize;
  bool fAllocated;

public:

#ifndef VECGEOM_NVCC

  VECGEOM_INLINE
  Array();

  VECGEOM_INLINE
  Array(const unsigned size);

  VECGEOM_INLINE
  Array(Array<Type> const &other);

  VECGEOM_INLINE
  ~Array();

  VECGEOM_INLINE
  void Allocate(const unsigned size);

  VECGEOM_INLINE
  void Deallocate();

#endif

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array(Type *const data, const unsigned size);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array& operator=(Array<Type> const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index) { return fData[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator[](const int index) const { return fData[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return fSize; }

public:

  typedef Iterator<Type> iterator;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> begin() { return Iterator<Type>(&fData[0]); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> end() { return Iterator<Type>(&fData[fSize]); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator<Type> begin() const { return ConstIterator<Type>(&fData[0]); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator<Type> end() const { return ConstIterator<Type>(&fData[fSize]); }

};

#ifndef VECGEOM_NVCC

template <typename Type>
Array<Type>::Array() : fSize(0), fAllocated(false) {}

template <typename Type>
Array<Type>::Array(const unsigned size) : fAllocated(true) {
  Allocate(size);
}

template <typename Type>
Array<Type>::Array(Array<Type> const &other) : fAllocated(true) {
  Allocate(other.fSize);
  std::copy(other.fData, other.fData+other.fSize, fData);
}

template <typename Type>
void Array<Type>::Allocate(const unsigned size) {
  Deallocate();
  fSize = size;
  fData = static_cast<Type*>(_mm_malloc(fSize*sizeof(Type),
                                        kAlignmentBoundary));
}

template <typename Type>
void Array<Type>::Deallocate() {
  if (fAllocated) {
    _mm_free(fData);
  } else {
    fData = NULL;
  }
  fSize = 0;
  fAllocated = false;
}

template <typename Type>
Array<Type>::~Array() {
  if (fAllocated) _mm_free(fData);
}

#endif

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Array<Type>::Array(Type *const data, const unsigned size)
    : fSize(size), fData(data), fAllocated(false) {}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Array<Type>& Array<Type>::operator=(Array<Type> const &other) {
  Deallocate();
  Allocate(other.fSize);
  copy(other.fData, other.fData+other.fSize, fData);
}

} // End global namespace

#endif // VECGEOM_BASE_ARRAY_H_