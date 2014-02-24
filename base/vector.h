#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include <vector>
#include "base/container.h"

namespace vecgeom {

template <typename Type>
class Vector : public Container<Type> {

private:

  std::vector<Type> vec;
  Type *begin_ptr, *end_ptr;
  int size_;

public:

  Vector() {
    begin_ptr = &vec[0];
    end_ptr = &vec[0];
    size_ = 0;
  }

  ~Vector() {}

  VECGEOM_INLINE
  Type& operator[](const int index) {
    return vec[index];
  }

  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return vec[index];
  }

  void push_back(Type const &item) {
    vec.push_back(item);
    size_ = vec.size();
    begin_ptr = &vec[0];
    end_ptr = &vec[size()];
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
    return VectorIterator(begin_ptr);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type> end() const {
    return VectorIterator(end_ptr);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual int size() const {
    return size_;
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_