/// \file Vector.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include "base/Global.h"

#include "base/Container.h"

namespace VECGEOM_NAMESPACE {

template <typename Type>
class Vector : public Container<Type> {

private:

  Type *vec_;
  int size_, memory_size_;
  bool allocated_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  Vector() : size_(0), memory_size_(1), allocated_(true) {
    vec_ = new Type[memory_size_];
  }

  VECGEOM_CUDA_HEADER_BOTH
  Vector(Type *const vec, const int size)
      : vec_(vec), size_(size), allocated_(false) {}

  ~Vector() {
    if (allocated_) delete[] vec_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& operator[](const int index) {
    return vec_[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& operator[](const int index) const {
    return vec_[index];
  }

  void push_back(const Type item) {
    if (size_ == memory_size_) {
      memory_size_ = memory_size_<<1;
      Type *vec_new = new Type[memory_size_];
      for (int i = 0; i < size_; ++i) vec_new[i] = vec_[i];
      delete[] vec_;
      vec_ = vec_new;
    }
    vec_[size_] = item;
    size_++;
  }

private:

  class VectorIterator : public Iterator<Type> {

  public:

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    VectorIterator(Type const *const e) : Iterator<Type>(e) {}

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    Iterator<Type>& operator++() {
      this->element_++;
      return *this;
    }

  };

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type> begin() const {
    return VectorIterator(&vec_[0]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type> end() const {
    return VectorIterator(&vec_[size_]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual int size() const {
    return size_;
  }

};

} // End global namespace

#endif // VECGEOM_BASE_CONTAINER_H_
