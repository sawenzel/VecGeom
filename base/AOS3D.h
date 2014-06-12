/// \file AOS3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_AOS3D_H_
#define VECGEOM_BASE_AOS3D_H_

#include "base/Global.h"

#include "base/TrackContainer.h"
#include "base/Vector3D.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

// #include <cassert>

namespace VECGEOM_NAMESPACE {

template <typename Type>
class AOS3D : public TrackContainer<Type> {

private:

  Vector3D<Type> *data_;

  typedef Vector3D<Type> VecType;

public:

  VECGEOM_CUDA_HEADER_BOTH
  AOS3D(Vector3D<Type> *const data, const int size);

  AOS3D(const int size);

  AOS3D(TrackContainer<Type> const &other);
  AOS3D(AOS3D const &other);
  AOS3D* operator=(AOS3D const &other);

  ~AOS3D();

  // Element access methods. Can be used to manipulate content.

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type> operator[](const int index) const {
    return data_[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type>& operator[](const int index) {
    return data_[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* data() { return data_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& x(const int index) const { return (data_[index])[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& y(const int index) const { return (data_[index])[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& z(const int index) const { return (data_[index])[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, const Type x, const Type y, const Type z);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const int index, Vector3D<Type> const &vec);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back(const Type x, const Type y, const Type z);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back(Vector3D<Type> const &vec);

  #ifdef VECGEOM_CUDA
  AOS3D<Type>* CopyToGpu(Vector3D<Type> *const data_gpu) const;
  #endif

};

template <typename Type>
AOS3D<Type>::AOS3D(Vector3D<Type> *const data, const int memory_size)
    : TrackContainer<Type>(memory_size, false), data_(data) {
  this->size_ = memory_size;
}

template <typename Type>
AOS3D<Type>::AOS3D(const int memory_size)
    : TrackContainer<Type>(memory_size, true) {
  data_ = static_cast<VecType*>(
            _mm_malloc(sizeof(VecType)*memory_size, kAlignmentBoundary)
          );
  for (int i = 0; i < memory_size; ++i) new(data_+i) VecType;
}

template <typename Type>
AOS3D<Type>::AOS3D(TrackContainer<Type> const &other)
    : TrackContainer<Type>(other.memory_size_, true) {
  this->size_ = other.size_;
  data_ = static_cast<VecType*>(
            _mm_malloc(sizeof(VecType)*other.size(), kAlignmentBoundary)
          );
  const int count = other.size();
  for (int i = 0; i < count; ++i) {
    data_[i] = other[i];
  }
}

template <typename Type>
AOS3D<Type>::~AOS3D() {
  if (this->allocated_) _mm_free(data_);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AOS3D<Type>::Set(const int index, const Type x, const Type y,
                      const Type z) {
  // assert(index < this->memory_size_);
  if (index >= this->size_) this->size_ = index+1;
  (data_[index])[0] = x;
  (data_[index])[1] = y;
  (data_[index])[2] = z;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AOS3D<Type>::Set(const int index, Vector3D<Type> const &vec) {
  // assert(index < this->memory_size_);
  if (index >= this->size_) this->size_ = index+1;
  data_[index] = vec;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AOS3D<Type>::push_back(const Type x, const Type y, const Type z) {
  // assert(this->size_ < this->memory_size_);
  (data_[this->size_])[0] = x;
  (data_[this->size_])[1] = y;
  (data_[this->size_])[2] = z;
  this->size_++;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AOS3D<Type>::push_back(Vector3D<Type> const &vec) {
  // assert(this->size_ < this->memory_size_);
  data_[this->size_] = vec;
  this->size_++;
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

template <typename Type> class Vector3D;
template <typename Type> class AOS3D;

AOS3D<Precision>* AOS3D_CopyToGpu(Vector3D<Precision> *const data,
                                  const int size);

template <typename Type>
AOS3D<Type>* AOS3D<Type>::CopyToGpu(
    Vector3D<Type> *const data_gpu) const {
  const int mem_size = this->size_*sizeof(Vector3D<Type>);
  vecgeom::CopyToGpu(data_, data_gpu, mem_size);
  return AOS3D_CopyToGpu(data_gpu, this->size_);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End namespace vecgeom

#endif // VECGEOM_BASE_AOS3D_H_
