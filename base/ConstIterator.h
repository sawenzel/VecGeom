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

protected:

  Type const *fElement;

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator(Type const *const e) : fElement(e) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator(ConstIterator<Type> const &other) : fElement(other.fElement) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator<Type>& operator=(ConstIterator<Type> const &other) {
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
  ConstIterator<Type> operator+(const int val) const {
    return ConstIterator<Type>(fElement + val);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ConstIterator<Type> operator-(const int val) const {
    return ConstIterator<Type>(fElement - val);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator*() const {
    return *fElement;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const* operator->() const {
    return fElement;
  }

  #define CONSTITERATOR_COMPARISON_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  bool operator OPERATOR(ConstIterator<Type> const &other) const { \
    return fElement OPERATOR other.fElement; \
  }
  CONSTITERATOR_COMPARISON_OP(<)
  CONSTITERATOR_COMPARISON_OP(>)
  CONSTITERATOR_COMPARISON_OP(<=)
  CONSTITERATOR_COMPARISON_OP(>=)
  CONSTITERATOR_COMPARISON_OP(!=)
  CONSTITERATOR_COMPARISON_OP(==)
  #undef ITERATOR_COMPARISON_OP

};

} // End global namespace

#endif // VECGEOM_BASE_CONSTITERATOR_H_