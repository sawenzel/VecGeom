/// \file Iterator.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ITERATOR_H_
#define VECGEOM_BASE_ITERATOR_H_

#include "base/Global.h"

#include <iterator>

namespace VECGEOM_NAMESPACE {

/// \brief Custom iterator class for use with container classes.
template <typename Type>
class Iterator : public std::iterator<std::forward_iterator_tag, Type> {

protected:

  Type *fElement;

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator(Type *const e) : fElement(e) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator(Iterator<Type> const &other) : fElement(other.fElement) {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type>& operator=(Iterator<Type> const &other) {
    fElement = other.fElement;
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator==(Iterator<Type> const &other) {
    return fElement == other.fElement;
  }

  virtual ~Iterator(){}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator!=(Iterator<Type> const &other) {
    return fElement != other.fElement;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type>& operator++() {
    fElement++;
    return *this;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type>& operator++(int) {
    Iterator<Type> temp(*this);
    ++(*this);
    return temp;
  }
#pragma GCC diagnostic pop

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> operator+(const int val) const {
    return Iterator<Type>(fElement + val);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Iterator<Type> operator-(const int val) const {
    return Iterator<Type>(fElement - val);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator*() {
    return *fElement;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* operator->() {
    return fElement;
  }

  #define ITERATOR_COMPARISON_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  bool operator OPERATOR(Iterator<Type> const &other) const { \
    return fElement OPERATOR other.fElement; \
  }
  ITERATOR_COMPARISON_OP(<)
  ITERATOR_COMPARISON_OP(>)
  ITERATOR_COMPARISON_OP(<=)
  ITERATOR_COMPARISON_OP(>=)
  ITERATOR_COMPARISON_OP(!=)
  ITERATOR_COMPARISON_OP(==)
  #undef ITERATOR_COMPARISON_OP

};

} // End global namespace

#endif // VECGEOM_BASE_ITERATOR_H_
