/**
 * @file iterator.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_ITERATOR_H_
#define VECGEOM_BASE_ITERATOR_H_

#include "base/global.h"
#include <iterator>

namespace vecgeom {

/**
 * @brief Custom iterator class for use with container classes.
 */
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

} // End namespace vecgeom

#endif // VECGEOM_BASE_ITERATOR_H_