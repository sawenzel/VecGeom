#ifndef VECGEOM_BASE_CONTAINER_H_
#define VECGEOM_BASE_CONTAINER_H_

namespace vecgeom {

template <typename Type>
class Container {

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& operator[](const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& operator[](const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual int Size() const =0;

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_