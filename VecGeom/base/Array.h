/// \file Array.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/backend/scalar/Backend.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename Type>
class Array : public AlignedBase {

private:
  Type *fData = nullptr;
  unsigned int fSize = 0;
  bool fAllocated = false;

public:
  Array() = default;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array(const unsigned int size);

  VECGEOM_FORCE_INLINE
  Array(Array<Type> const &other);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array(Type *data, unsigned int size);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ~Array();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array &operator=(Array<Type> const &other);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &operator[](const int index) { return fData[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &operator[](const int index) const { return fData[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int size() const { return fSize; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Allocate(const unsigned int size);

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Deallocate();

  typedef Type *iterator;
  typedef Type const *const_iterator;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type *begin() { return &fData[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type *end() { return &fData[fSize]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const *cbegin() const { return &fData[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const *cend() const { return &fData[fSize]; }
};

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Array<Type>::Array(const unsigned int initSize)
{
  Allocate(initSize);
}

template <typename Type>
Array<Type>::Array(Array<Type> const &other)
{
  Allocate(other.fSize);
  copy(other.fData, other.fData + other.fSize, fData);
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
Array<Type>::Array(Type *data, unsigned int initSize) : fData(data), fSize(initSize), fAllocated(false)
{
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Array<Type>::~Array()
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (fAllocated) vecCore::AlignedFree(fData);
#endif
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
void Array<Type>::Allocate(const unsigned int initSize)
{
  Deallocate();
  fSize = initSize;
  if (initSize == 0) return;

#ifndef VECCORE_CUDA
  fData = static_cast<Type *>(vecCore::AlignedAlloc(kAlignmentBoundary, fSize * sizeof(Type)));
#else
  fData = static_cast<Type *>(malloc(fSize * sizeof(Type))); // new Type[fSize];
#endif
  fAllocated = true;
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
void Array<Type>::Deallocate()
{
  if (fAllocated) {
#ifndef VECCORE_CUDA
    vecCore::AlignedFree(fData);
#else
    free(fData);
#endif
  }
  fData      = nullptr;
  fSize      = 0;
  fAllocated = false;
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
Array<Type> &Array<Type>::operator=(Array<Type> const &other)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  Deallocate();
  Allocate(other.fSize);
  copy(other.fData, other.fData + other.fSize, fData);
#else
  fData      = other.fData;
  fSize      = other.fSize;
  fAllocated = false;
#endif
  return *this;
}
}
} // End global namespace

#endif // VECGEOM_BASE_ARRAY_H_
