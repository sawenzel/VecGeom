/// \file AOS3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_AOS3D_H_
#define VECGEOM_BASE_AOS3D_H_

#include "base/Global.h"

#include "base/Container3D.h"
#include "base/Vector3D.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( template <typename Type> class AOS3D; )

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
class AOS3D : Container3D<AOS3D<T> > {

private:

  bool fAllocated;
  size_t fSize, fCapacity;
  Vector3D<T> *fContent;

  typedef Vector3D<T> Vec_t;

public:

  typedef T value_type;

  VECGEOM_CUDA_HEADER_BOTH
  AOS3D(Vector3D<T> *data, size_t size);

  AOS3D(size_t size);

  AOS3D(AOS3D<T> const &other);

  AOS3D& operator=(AOS3D<T> const &other);

  ~AOS3D();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  size_t size() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  size_t capacity() const;

  VECGEOM_INLINE
  void resize(size_t newSize);

  VECGEOM_INLINE
  void reserve(size_t newCapacity);

  VECGEOM_INLINE
  void clear();  

  // Element access methods. Can be used to manipulate content.

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<T> operator[](size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<T>& operator[](size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T* content();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T const* content() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T x(size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T& x(size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T y(size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T& y(size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T z(size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T& z(size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set(size_t index, T x, T y, T z);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set(size_t index, Vector3D<T> const &vec);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back(T x, T y, T z);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back(Vector3D<T> const &vec);

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::AOS3D<T> > CopyToGpu(DevicePtr<cuda::Vector3D<T> > contentGpu) const;
#endif

private:

  void Deallocate();

};

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
AOS3D<T>::AOS3D(Vector3D<T> *content, size_t size)
    : fAllocated(false), fSize(size), fCapacity(fSize), fContent(content) {}

template <typename T>
AOS3D<T>::AOS3D(size_t capacity)
    : fAllocated(true), fSize(0), fCapacity(capacity), fContent(NULL) {
  reserve(fCapacity);
}

template <typename T>
AOS3D<T>::AOS3D(AOS3D<T> const &rhs)
    : fAllocated(rhs.fAllocated), fSize(rhs.fSize),
      fCapacity(rhs.fCapacity), fContent(NULL) {
  *this = rhs;
}

template <typename T>
AOS3D<T>& AOS3D<T>::operator=(AOS3D<T> const &rhs) {
  clear();
  if (rhs.fAllocated) {
    reserve(rhs.fCapacity);
    copy(rhs.fContent, rhs.fContent+rhs.fSize, fContent);
  } else {
    fContent = rhs.fContent;
    fAllocated = false;
    fCapacity = rhs.fCapacity;
  }
  fSize = rhs.fSize;
  return *this;
}

template <typename T>
AOS3D<T>::~AOS3D() {
  Deallocate();
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
size_t AOS3D<T>::size() const { return fSize; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
size_t AOS3D<T>::capacity() const { return fCapacity; }

template <typename T>
void AOS3D<T>::resize(size_t newSize) {
  Assert(newSize <= fCapacity);
  fSize = newSize;
}

template <typename T>
void AOS3D<T>::reserve(size_t newCapacity) {
  fCapacity = newCapacity;
  Vec_t *contentNew;
#ifndef VECGEOM_NVCC
  contentNew = static_cast<Vec_t*>(_mm_malloc(sizeof(Vec_t)*fCapacity,
                                   kAlignmentBoundary));
#else
  contentNew = new Vec_t[fCapacity];
#endif
  fSize = (fSize > fCapacity) ? fCapacity : fSize;
  if (fContent) {
    copy(fContent, fContent+fSize, contentNew);
  }
  Deallocate();
  fContent = contentNew;
  fAllocated = true;
}

template <typename T>
void AOS3D<T>::clear() {
  Deallocate();
  fAllocated = false;
  fSize = 0;
  fCapacity = 0;
}

template <typename T>
void AOS3D<T>::Deallocate() {
  if (fAllocated) {
#ifndef VECGEOM_NVCC
    _mm_free(fContent);
#else
    delete fContent;
#endif
  }
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<T> AOS3D<T>::operator[](size_t index) const {
  return fContent[index];
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<T>& AOS3D<T>::operator[](size_t index) {
  return fContent[index];
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T* AOS3D<T>::content() { return fContent; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T AOS3D<T>::x(size_t index) const { return (fContent[index])[0]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T& AOS3D<T>::x(size_t index) { return (fContent[index])[0]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T AOS3D<T>::y(size_t index) const { return (fContent[index])[1]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T& AOS3D<T>::y(size_t index) { return (fContent[index])[1]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T AOS3D<T>::z(size_t index) const { return (fContent[index])[2]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T& AOS3D<T>::z(size_t index) { return (fContent[index])[2]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void AOS3D<T>::set(size_t index, T x, T y, T z) {
  (fContent[index])[0] = x;
  (fContent[index])[1] = y;
  (fContent[index])[2] = z;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void AOS3D<T>::set(size_t index, Vector3D<T> const &vec) {
  fContent[index] = vec;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void AOS3D<T>::push_back(T x, T y, T z) {
  (fContent[fSize])[0] = x;
  (fContent[fSize])[1] = y;
  (fContent[fSize])[2] = z;
  ++fSize;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void AOS3D<T>::push_back(Vector3D<T> const &vec) {
  fContent[fSize] = vec;
  ++fSize;
}


#ifdef VECGEOM_CUDA_INTERFACE

template <typename T>
DevicePtr< cuda::AOS3D<T> > AOS3D<T>::CopyToGpu(DevicePtr<cuda::Vector3D<T> > contentGpu) const {
   contentGpu.ToDevice(fContent, fSize);

   DevicePtr< cuda::AOS3D<T> > gpu_ptr;
   gpu_ptr.Allocate();
   gpu_ptr.Construct(contentGpu, fSize);
}

#endif // VECGEOM_CUDA_INTERFACE

} } // End namespace vecgeom


#endif // VECGEOM_BASE_AOS3D_H_
