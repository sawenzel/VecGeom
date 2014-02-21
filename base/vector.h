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

public:

  Vector() {
    begin_ptr = &vec[0];
    end_ptr = &vec[0];
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

  VECGEOM_INLINE
  int size() const {
    return vec.size();
  }

  void push_back(Type const &item) {
    vec.push_back(item);
    begin_ptr = &vec[0];
    end_ptr = &vec[size()-1];
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

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_