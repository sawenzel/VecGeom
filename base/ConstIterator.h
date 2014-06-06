/// \file ConstIterator.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_CONSTITERATOR_H_
#define VECGEOM_BASE_CONSTITERATOR_H_

#include "base/Global.h"

#include <iterator>

namespace VECGEOM_NAMESPACE {

/// \brief Custom iterator class for use with container classes.
template <typename Type>
class ConstIterator : public std::iterator<std::forward_iterator_tag, Type> {

private:

  Type const *fElement;

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator(Type *const e) : fElement(e) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator(ConstIterator const &other) : fElement(other.fElement) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator& operator=(ConstIterator<Type> const &other) {
    fElement = other.fElement;
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator==(ConstIterator<Type> const &other) {
    return fElement == other.fElement;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator!=(ConstIterator<Type> const &other) {
    return fElement != other.fElement;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator<Type>& operator++() {
    this->fElement++;
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator<Type>& operator++(int) {
    ConstIterator<Type> temp(*this);
    ++(*this);
    return temp;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator*() {
    return *fElement;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const* operator->() {
    return fElement;
  }

};

} // End global namespace

#endif // VECGEOM_BASE_CONSTITERATOR_H_