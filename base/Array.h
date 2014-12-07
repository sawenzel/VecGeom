/// \file Array.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "base/Global.h"

#include "base/AlignedBase.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename Type>
class Array : public AlignedBase {

private:

  Type *fData;
  int fSize;
  bool fAllocated;

public:

  VECGEOM_INLINE
  Array();

  VECGEOM_INLINE
  Array(const unsigned size);

  VECGEOM_INLINE
  Array(Array<Type> const &other);

  VECGEOM_INLINE
  ~Array();

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

  VECGEOM_INLINE
  void Allocate(const unsigned size);

  VECGEOM_INLINE
  void Deallocate();

public:

  typedef Type* iterator;
  typedef Type const* const_iterator;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* begin() { return &fData[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* end() { return &fData[fSize]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const* cbegin() const { return &fData[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const* cend() const { return &fData[fSize]; }

};

template <typename Type>
Array<Type>::Array() : fData(NULL), fSize(0), fAllocated(false) {}

template <typename Type>
Array<Type>::Array(const unsigned size) : fData(NULL), fAllocated(true) {
  Allocate(size);
}

template <typename Type>
Array<Type>::Array(Array<Type> const &other) : fData(NULL), fAllocated(true) {
  Allocate(other.fSize);
  copy(other.fData, other.fData+other.fSize, fData);
}

template <typename Type>
Array<Type>::~Array() {
#ifndef VECGEOM_CUDA
  if (fAllocated) _mm_free(fData);
#else
  if (fAllocated) delete fData;
#endif
}

template <typename Type>
void Array<Type>::Allocate(const unsigned size) {
  Deallocate();
  fSize = size;
#ifndef VECGEOM_NVCC
  fData = static_cast<Type*>(_mm_malloc(fSize*sizeof(Type),
                                        kAlignmentBoundary));
#else
  fData = new Type[fSize];
#endif
}

template <typename Type>
void Array<Type>::Deallocate() {
  if (fAllocated) {
#ifndef VECGEOM_NVCC
    _mm_free(fData);
#else
    delete fData;
#endif
  } else {
    fData = NULL;
  }
  fSize = 0;
  fAllocated = false;
}

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
  return *this;
}

} } // End global namespace

#endif // VECGEOM_BASE_ARRAY_H_
