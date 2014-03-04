#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "base/global.h"
#include "base/iterator.h"

namespace vecgeom {

template <typename Type>
class Array : public Container<Type> {

private:

  Type *arr;

  int size_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  Array(const int size) {
    size_ = size;
    arr = new Type[size];
  }

  VECGEOM_CUDA_HEADER_BOTH
  Array(Type *const arr_, const int size__) {
    arr = arr_;
    size_ = size__;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index) {
    return arr[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return arr[index];
  }

private:

  class ArrayIterator : public Iterator<Type> {

  public:

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    ArrayIterator(Type const *const e) : Iterator<Type>(e) {}

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    virtual Iterator<Type>& operator++() {
      this->element_++;
      return *this;
    }

  };

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type> begin() const {
    return ArrayIterator(&arr[0]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type> end() const {
    return ArrayIterator(&arr[size()]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual int size() const {
    return size_;
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_ARRAY_H_