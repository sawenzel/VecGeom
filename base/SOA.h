/// \file SOA.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_SOA_H_
#define VECGEOM_BASE_SOA_H_

#include "base/Global.h"

#include "base/AlignedBase.h"

namespace VECGEOM_NAMESPACE { 

template <typename T, int rows, int columns>
struct SOAData : public AlignedBase {
  T fHead[rows] VECGEOM_ALIGNED;
  SOAData<T, rows, columns-1> fTail;
};
template <typename T, int rows>
struct SOAData<T, rows, 0> : public AlignedBase {};

template <typename T, int columns, int rows>
class SOA : public AlignedBase {

private:

  SOAData<T, rows, columns> fData;

  static VECGEOM_CONSTEXPR unsigned long fgColumnSize
      = sizeof(SOA<T, columns, rows>)/columns;

public:

  typedef T Column_t[rows];

  SOA() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Column_t& operator[](int index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Column_t const& operator[](int index) const;

};

template <typename T, int columns, int rows>
VECGEOM_CUDA_HEADER_BOTH
typename SOA<T, columns, rows>::Column_t&
SOA<T, columns, rows>::operator[](int index) {
  return *(&fData.fHead + index*fgColumnSize);
}

template <typename T, int columns, int rows>
VECGEOM_CUDA_HEADER_BOTH
typename SOA<T, columns, rows>::Column_t const&
SOA<T, columns, rows>::operator[](int index) const {
  return *(&fData.fHead + index*fgColumnSize);
}

} // End global namespace

#endif // VECGEOM_BASE_SOA_H_