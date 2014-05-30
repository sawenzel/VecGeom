/**
 * @file array.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_ARRAY_H_
#define VECGEOM_BASE_ARRAY_H_

#include "base/global.h"
#include "base/iterator.h"

namespace VECGEOM_NAMESPACE {

template <typename Type>
class Array : public Container<Type> {

private:

  Type *arr_;
  int size_;
  bool allocated;

public:

  VECGEOM_CUDA_HEADER_BOTH
  Array(const int size) : size_(size), allocated(true) {
    arr_ = new Type[size_];
  }

  VECGEOM_CUDA_HEADER_BOTH
  Array(Type *const arr, const int size)
      : arr_(arr), size_(size), allocated(false) {}

  ~Array() { if (allocated) delete arr_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& operator[](const int index) {
    return arr_[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& operator[](const int index) const {
    return arr_[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual int size() const {
    return size_;
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
    return ArrayIterator(&arr_[0]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type> end() const {
    return ArrayIterator(&arr_[size()]);
  }

};

} // End global namespace

#endif // VECGEOM_BASE_ARRAY_H_