#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "base/global.h"
#include "base/track_container.h"

namespace vecgeom {

template <typename Type>
class SOA3D : public TrackContainer<Type> {

private:

  Type *x_, *y_, *z_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(Type *const x, Type *const y, Type *const z, const unsigned size)
      : TrackContainer<Type>(size, true), x_(x), y_(y), z_(z) {
    TrackContainer<Type>(size, false);
  }

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D() {
    SOA3D(NULL, NULL, NULL, 0);
  }

  SOA3D(const unsigned size) : TrackContainer<Type>(size, true) {
    x_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
    y_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
    z_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*size, kAlignmentBoundary));
  }

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(TrackContainer<Type> const &other);

  ~SOA3D() {
    if (this->allocated_) {
      _mm_free(x_);
      _mm_free(y_);
      _mm_free(z_);
    }
  }

  /**
   * Constructs a vector across all three coordinates from the given index. 
   */ 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(x_[index], y_[index], z_[index]);
  }

  // Element access methods.
  // Can be used to manipulate content if necessary.

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& x(const int index) { return x_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& x(const int index) const { return x_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& y(const int index) { return y_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& y(const int index) const { return y_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& z(const int index) { return z_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& z(const int index) const { return z_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void Set(const int index, const Type x, const Type y, const Type z) {
    x_[index] = x;
    y_[index] = y;
    z_[index] = z;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void Set(const int index, Vector3D<Type> const &vec) {
    x_[index] = vec[0];
    y_[index] = vec[1];
    z_[index] = vec[2];
  }


  #ifdef VECGEOM_CUDA

  /**
   * Allocates and copies the data of this SOA to the GPU, then creates and
   * returns a new SOA object that points to GPU memory.
   */
  VECGEOM_CUDA_HEADER_HOST
  SOA3D<Type> CopyToGpu() const;

  /**
   * Only works for SOA pointing to the GPU.
   */
  VECGEOM_CUDA_HEADER_HOST
  void FreeFromGpu();

  #endif // VECGEOM_CUDA

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_SOA3D_H_