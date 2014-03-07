#ifndef VECGEOM_BASE_AOS3D_H_
#define VECGEOM_BASE_AOS3D_H_

#include "base/global.h"
#include "base/track_container.h"

namespace vecgeom {

template <typename Type>
class AOS3D : protected TrackContainer<Type> {

private:

  Vector3D<Type> *data_;

  typedef Vector<Type> VecType;

public:

  AOS3D(const int size) {
    TrackContainer<Type>(size, true);
    data_ = static_cast<VecType*>(
              _mm_malloc(sizeof(VecType)*size, kAlignmentBoundary)
            );
    for (int i = 0; i < size; ++i) new(data_+i) VecType;
  }

  VECGEOM_CUDA_HEADER_BOTH
  AOS3D(TrackContainer<Type> const &other);

  ~AOS3D() {
    if (this->allocated_) _mm_free(data_);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Vector3D<Type> operator[](const int index) const {
    return data_[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type>& operator[](const int index) {
    return data_[index];
  }

  // Element access methods.
  // Can be used to manipulate content if necessary.

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
  virtual void Set(const int index, const Type x, const Type y, const Type z) {
    (data_[index])[0] = x;
    (data_[index])[1] = y;
    (data_[index])[2] = z;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void Set(const int index, Vector3D<Type> const &vec) {
    data_[index] = vec;
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_AOS3D_H_