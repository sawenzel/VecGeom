/**
 * @file aos3d.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_AOS3D_H_
#define VECGEOM_BASE_AOS3D_H_

#include "base/global.h"

#include "base/track_container.h"
#include "base/vector3d.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/interface.h"
#endif

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
  virtual Type& x(const int index) { return (data_[index])[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& x(const int index) const { return (data_[index])[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& y(const int index) { return (data_[index])[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& y(const int index) const { return (data_[index])[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& z(const int index) { return (data_[index])[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& z(const int index) const { return (data_[index])[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void Set(const int index, const Type x, const Type y, const Type z);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void Set(const int index, Vector3D<Type> const &vec);

  #ifdef VECGEOM_CUDA
  AOS3D<Type>* CopyToGpu(Vector3D<Type> *const data_gpu) const;
  #endif

};

template <typename Type>
AOS3D<Type>::AOS3D(Vector3D<Type> *const data, const int size)
    : TrackContainer<Type>(size, false), data_(data) {}

template <typename Type>
AOS3D<Type>::AOS3D(const int size) : TrackContainer<Type>(size, true){
  data_ = static_cast<VecType*>(
            _mm_malloc(sizeof(VecType)*size, kAlignmentBoundary)
          );
  for (int i = 0; i < size; ++i) new(data_+i) VecType;
}

template <typename Type>
AOS3D<Type>::AOS3D(TrackContainer<Type> const &other)
    : TrackContainer<Type>(other.size_, true) {
  data_ = static_cast<VecType*>(
            _mm_malloc(sizeof(VecType)*other.size(), kAlignmentBoundary)
          );
  const unsigned count = other.size();
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
  (data_[index])[0] = x;
  (data_[index])[1] = y;
  (data_[index])[2] = z;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AOS3D<Type>::Set(const int index, Vector3D<Type> const &vec) {
  data_[index] = vec;
}

} // End global namespace

namespace vecgeom {

template <typename Type> class Vector3D;
template <typename Type> class AOS3D;

AOS3D<Precision>* AOS3DCopyToGpuInterface(Vector3D<Precision> *const data,
                                          const unsigned size);

#ifdef VECGEOM_CUDA_INTERFACE

template <typename Type>
AOS3D<Type>* AOS3D<Type>::CopyToGpu(Vector3D<Type> *const data_gpu) const {
  const int count = this->size();
  const int mem_size = count*sizeof(Vector3D<Type>);
  vecgeom::CopyToGpu(data_, data_gpu, mem_size);
  return AOS3DCopyToGpuInterface(data_gpu, count);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End namespace vecgeom

#endif // VECGEOM_BASE_AOS3D_H_