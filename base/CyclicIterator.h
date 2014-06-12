/// \file CyclicIterator.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_CYCLICITERATOR_H_
#define VECGEOM_BASE_CYCLICITERATOR_H_

#include "base/Global.h"

namespace VECGEOM_NAMESPACE {

namespace {

// In order to avoid code duplication, traits are used to derive the pointer and
// reference type of the elements iterated over by CyclicIterator to be const or
// non-const.

template <typename TypeT, bool IsConstT>
struct CyclicIteratorTraits {};

template <typename TypeT>
struct CyclicIteratorTraits<TypeT, true> {
  typedef TypeT const* Pointer_t;
  typedef TypeT const& Reference_t;
};

template <typename TypeT>
struct CyclicIteratorTraits<TypeT, false> {
  typedef TypeT* Pointer_t;
  typedef TypeT& Reference_t;
};

}

/// \brief Iterator that is cyclic when plus and minus operators are applied.
///        This does not apply for incrementation operators, which will
///        work as usual to allow proper iteration through the sequence.
template <typename TypeT, bool IsConstT>
class CyclicIterator : std::iterator<std::bidirectional_iterator_tag, TypeT> {

private:

  typedef typename CyclicIteratorTraits<TypeT, IsConstT>::Pointer_t Pointer_t;
  typedef typename CyclicIteratorTraits<TypeT, IsConstT>::Reference_t
      Reference_t;
  typedef CyclicIterator<TypeT, IsConstT> CyclicIterator_t;

  Pointer_t fBegin, fEnd, fCurrent;

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator(const Pointer_t begin, const Pointer_t end,
                 const Pointer_t current);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator(CyclicIterator_t const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t& operator=(CyclicIterator_t const &rhs);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t& operator++();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t operator++(int);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t& operator--();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t operator--(int);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t operator+(const int val) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  CyclicIterator_t operator-(const int val) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Reference_t operator*() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Pointer_t operator->() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator==(CyclicIterator_t const &other) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator!=(CyclicIterator_t const &other) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  operator Pointer_t();

};

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT>::CyclicIterator(
    const Pointer_t begin, const Pointer_t end, const Pointer_t current)
    : fBegin(begin), fEnd(end), fCurrent(current) {}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT>::CyclicIterator(CyclicIterator_t const &other)
    : fBegin(other.fBegin), fEnd(other.fEnd), fCurrent(other.fCurrent) {}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT>& CyclicIterator<TypeT, IsConstT>::operator=(
    CyclicIterator_t const &rhs) {
  fBegin = rhs.fBegin;
  fEnd = rhs.fEnd;
  fCurrent = rhs.fCurrent;
  return *this;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT>& CyclicIterator<TypeT, IsConstT>::operator++() {
  ++fCurrent;
  return *this;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT> CyclicIterator<TypeT, IsConstT>::operator++(
    int) {
  CyclicIterator_t temp(*this);
  ++(*this);
  return temp;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT>& CyclicIterator<TypeT, IsConstT>::operator--() {
  --fCurrent;
  return *this;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT> CyclicIterator<TypeT, IsConstT>::operator--(
    int) {
  CyclicIterator_t temp(*this);
  --(*this);
  return temp;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT> CyclicIterator<TypeT, IsConstT>::operator+(
    const int val) const {
  return CyclicIterator<TypeT, IsConstT>(
             fBegin, fEnd, fBegin + ((fEnd-fCurrent) + val) % (fEnd-fBegin));
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT> CyclicIterator<TypeT, IsConstT>::operator-(
    const int val) const {
  return CyclicIterator<TypeT, IsConstT>(
             fBegin, fEnd, fBegin + ((fEnd-fCurrent) - val) % (fEnd-fBegin));
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
typename CyclicIterator<TypeT, IsConstT>::Reference_t
    CyclicIterator<TypeT, IsConstT>::operator*() const {
  return *fCurrent;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
typename CyclicIterator<TypeT, IsConstT>::Pointer_t
    CyclicIterator<TypeT, IsConstT>::operator->() const {
  return fCurrent;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
bool CyclicIterator<TypeT, IsConstT>::operator==(
    CyclicIterator_t const &other) const {
  return fCurrent == other.fCurrent;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
bool CyclicIterator<TypeT, IsConstT>::operator!=(
    CyclicIterator_t const &other) const {
  return fCurrent != other.fCurrent;
}

template <typename TypeT, bool IsConstT>
VECGEOM_CUDA_HEADER_BOTH
CyclicIterator<TypeT, IsConstT>::operator
    CyclicIterator<TypeT, IsConstT>::Pointer_t() {
  return fCurrent;
}

} // End global namespace

#endif // VECGEOM_BASE_CYCLICITERATOR_H_