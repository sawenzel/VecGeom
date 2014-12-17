/// \file SOA3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "base/Global.h"

#include "base/Container3D.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( template <typename Type> class SOA3D; )

inline namespace VECGEOM_IMPL_NAMESPACE {

// gcc 4.8.2's -Wnon-virtual-dtor is broken and turned on by -Weffc++, we
// need to disable it for SOA3D

#if __GNUC__ < 3 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 8)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Weffc++"
#define GCC_DIAG_POP_NEEDED
#endif

template <typename T>
class SOA3D : Container3D<SOA3D<T> > {

private:

  bool fAllocated;
  size_t fSize, fCapacity;
  T *fX, *fY, *fZ;

public:

  typedef T value_type;

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(T *x, T *y, T *z, size_t size);

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(size_t size);

  SOA3D(SOA3D<T> const &other);

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D();

  VECGEOM_CUDA_HEADER_BOTH
  SOA3D& operator=(SOA3D<T> const &other);

  VECGEOM_CUDA_HEADER_BOTH
  ~SOA3D();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  size_t size() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  size_t capacity() const;

  VECGEOM_INLINE
  void resize(size_t newSize);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void reserve(size_t newCapacity);

  VECGEOM_INLINE
  void clear();

  // Element access methods

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<T> operator[](size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T x(size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T& x(size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T* x();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T const* x() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T y(size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T& y(size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T* y();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T const* y() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T z(size_t index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T& z(size_t index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T* z();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  T const* z() const;

  // Element manipulation methods

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
  DevicePtr< cuda::SOA3D<T> > CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu) const;
  DevicePtr< cuda::SOA3D<T> > CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu, size_t size) const;
#endif // VECGEOM_CUDA

private:

  VECGEOM_CUDA_HEADER_BOTH
  void Allocate();

  VECGEOM_CUDA_HEADER_BOTH
  void Deallocate();

};

#if defined(GCC_DIAG_POP_NEEDED)

#pragma GCC diagnostic pop
#undef GCC_DIAG_POP_NEEDED

#endif

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<T>::SOA3D(T *xval, T *yval, T *zval, size_t sz)
    : fAllocated(false), fSize(sz), fCapacity(fSize), fX(xval), fY(yval), fZ(zval) {}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<T>::SOA3D(size_t sz)
    : fAllocated(true), fSize(0), fCapacity(sz),
      fX(NULL), fY(NULL), fZ(NULL) {
  Allocate();
}

template <typename T>
SOA3D<T>::SOA3D(SOA3D<T> const &rhs)
    : fAllocated(false), fSize(rhs.fSize),
      fCapacity(rhs.fCapacity), fX(NULL), fY(NULL), fZ(NULL) {
  if (rhs.fAllocated) {
    Allocate();
    copy(rhs.fX, rhs.fX+rhs.fSize, fX);
    copy(rhs.fY, rhs.fY+rhs.fSize, fY);
    copy(rhs.fZ, rhs.fZ+rhs.fSize, fZ);
  } else {
    fX = rhs.fX;
    fY = rhs.fY;
    fZ = rhs.fZ;
  }
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<T>::SOA3D()
    : fAllocated(false), fSize(0), fCapacity(0), fX(NULL), fY(NULL), fZ(NULL) {}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<T>& SOA3D<T>::operator=(SOA3D<T> const &rhs) {
#ifndef VECGEOM_NVCC_DEVICE
  fSize = rhs.fSize;
  fCapacity = rhs.fCapacity;
  Deallocate();
  if (rhs.fAllocated) {
    Allocate();
    copy(rhs.fX, rhs.fX+rhs.fSize, fX);
    copy(rhs.fY, rhs.fY+rhs.fSize, fY);
    copy(rhs.fZ, rhs.fZ+rhs.fSize, fZ);
  } else {
    fX = rhs.fX;
    fY = rhs.fY;
    fZ = rhs.fZ;
  }
#else
  fAllocated = false;
  fSize = rhs.fSize;
  fCapacity = rhs.fCapacity;
  fX = rhs.fX;
  fY = rhs.fY;
  fZ = rhs.fZ;
#endif
  return *this;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<T>::~SOA3D() {
#ifndef VECGEOM_NVCC_DEVICE
  Deallocate();
#endif
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
size_t SOA3D<T>::size() const { return fSize; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
size_t SOA3D<T>::capacity() const { return fCapacity; }

template <typename T>
void SOA3D<T>::resize(size_t newSize) {
  Assert(newSize <= fCapacity);
  fSize = newSize;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::reserve(size_t newCapacity) {
  fCapacity = newCapacity;
  T *xNew, *yNew, *zNew;
  xNew = AlignedAllocate<T>(fCapacity);
  yNew = AlignedAllocate<T>(fCapacity);
  zNew = AlignedAllocate<T>(fCapacity);
  fSize = (fSize > fCapacity) ? fCapacity : fSize;
  if (fX && fY && fZ) {
    copy(fX, fX+fSize, xNew);
    copy(fY, fY+fSize, yNew);
    copy(fZ, fZ+fSize, zNew);
  }
  Deallocate();
  fX = xNew;
  fY = yNew;
  fZ = zNew;
  fAllocated = true;
}

template <typename T>
void SOA3D<T>::clear() {
  Deallocate();
  fSize = 0;
  fCapacity = 0;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::Allocate() {
  fX = AlignedAllocate<T>(fCapacity);
  fY = AlignedAllocate<T>(fCapacity);
  fZ = AlignedAllocate<T>(fCapacity);
  fAllocated = true;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::Deallocate() {
  if (fAllocated) {
    AlignedFree(fX);
    AlignedFree(fY);
    AlignedFree(fZ);
  }
  fAllocated = false;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<T> SOA3D<T>::operator[](size_t index) const {
  return Vector3D<T>(fX[index], fY[index], fZ[index]);
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T SOA3D<T>::x(size_t index) const { return fX[index]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T& SOA3D<T>::x(size_t index) { return fX[index]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T* SOA3D<T>::x() { return fX; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T const* SOA3D<T>::x() const { return fX; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T SOA3D<T>::y(size_t index) const { return fY[index]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T& SOA3D<T>::y(size_t index) { return fY[index]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T* SOA3D<T>::y() { return fY; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T const* SOA3D<T>::y() const { return fY; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T SOA3D<T>::z(size_t index) const { return fZ[index]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T& SOA3D<T>::z(size_t index) { return fZ[index]; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T* SOA3D<T>::z() { return fZ; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
T const* SOA3D<T>::z() const { return fZ; }

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::set(size_t index, T xval, T yval, T zval) {
  fX[index] = xval;
  fY[index] = yval;
  fZ[index] = zval;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::set(size_t index, Vector3D<T> const &vec) {
  fX[index] = vec[0];
  fY[index] = vec[1];
  fZ[index] = vec[2];
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::push_back(T xval, T yval, T zval) {
  fX[fSize] = xval;
  fY[fSize] = yval;
  fZ[fSize] = zval;
  ++fSize;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
void SOA3D<T>::push_back(Vector3D<T> const &vec) {
  push_back(vec[0], vec[1], vec[2]);
}

#ifdef VECGEOM_CUDA_INTERFACE

template <typename T>
DevicePtr< cuda::SOA3D<T> > SOA3D<T>::CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu) const {
   xGpu.ToDevice(fX, fSize);
   yGpu.ToDevice(fY, fSize);
   zGpu.ToDevice(fZ, fSize);

   DevicePtr< cuda::SOA3D<T> > gpu_ptr;
   gpu_ptr.Allocate();
   gpu_ptr.Construct(xGpu, yGpu, zGpu, fSize);
}

template <typename T>
DevicePtr< cuda::SOA3D<T> > SOA3D<T>::CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu, size_t count) const {
   xGpu.ToDevice(fX, count);
   yGpu.ToDevice(fY, count);
   zGpu.ToDevice(fZ, count);

   DevicePtr< cuda::SOA3D<T> > gpu_ptr;
   gpu_ptr.Allocate();
   gpu_ptr.Construct(xGpu, yGpu, zGpu, fSize);
}

#endif // VECGEOM_CUDA

} } // End global namespace

#endif // VECGEOM_BASE_SOA3D_H_
