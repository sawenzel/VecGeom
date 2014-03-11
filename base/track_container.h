#ifndef VECGEOM_BASE_TRACKCONTAINER_H_
#define VECGEOM_BASE_TRACKCONTAINER_H_

#include "base/global.h"

namespace vecgeom {

template <typename Type>
class TrackContainer {

private:

  int size_;

protected:

  bool allocated_;

  TrackContainer(const unsigned size, const bool allocated)
      : size_(size), allocated_(allocated) {}

  virtual ~TrackContainer() {}

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return size_; }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Vector3D<Type> operator[](const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type& x(const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type const& x(const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type& y(const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type const& y(const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type& z(const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type const& z(const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Set(const int index, const Type x, const Type y,
                   const Type z) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Set(const int index, Vector3D<Type> const &vec) =0;

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_TRACKCONTAINER_H_