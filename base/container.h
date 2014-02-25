#ifndef VECGEOM_BASE_CONTAINER_H_
#define VECGEOM_BASE_CONTAINER_H_

#include "base/utilities.h"

namespace vecgeom {

template <typename Type>
class Iterator : public std::iterator<std::forward_iterator_tag, Type> {

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator(Type const *const e) : element_(e) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator(Iterator const &other) : element_(other.element_) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator& operator=(Iterator const &other) {
    element_ = other.element_;
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator==(Iterator const &other) {
    return element_ == other.element_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator!=(Iterator const &other) {
    return element_ != other.element_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator& operator++() {
    this->element_++;
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator& operator++(int) {
    Iterator temp(*this);
    ++(*this);
    return temp;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator*() {
    return *element_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const* operator->() {
    return element_;
  }

protected:

  Type const *element_;

};

template <typename Type>
class Container {

public:

  Container() {}

  virtual ~Container() {}

  VECGEOM_CUDA_HEADER_BOTH
  virtual Iterator<Type> begin() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Iterator<Type> end() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual int size() const =0;

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_