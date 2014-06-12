/// \file Iterator.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ITERATOR_H_
#define VECGEOM_BASE_ITERATOR_H_

#include "base/Global.h"

#include <iterator>

namespace VECGEOM_NAMESPACE {

/**
 * @brief Custom iterator class for use with container classes.
 */
template <typename Type>
class Iterator : public ::std::iterator< ::std::forward_iterator_tag, Type> {

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator(Type const *const e) : element_(e) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator(Iterator<Type> const &other) : element_(other.element_) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type>& operator=(Iterator<Type> const &other) {
    element_ = other.element_;
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator==(Iterator<Type> const &other) {
    return element_ == other.element_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator!=(Iterator<Type> const &other) {
    return element_ != other.element_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Iterator<Type>& operator++() {
    this->element_++;
    return *this;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> operator++(int) {
    Iterator<Type> temp(*this);
    ++(*this);
    return temp;
  }
#pragma GCC diagnostic pop

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

} // End global namespace

#endif // VECGEOM_BASE_ITERATOR_H_
