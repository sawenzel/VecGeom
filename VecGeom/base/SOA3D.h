/// \file SOA3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/base/Container3D.h"
#include "VecGeom/backend/scalar/Backend.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#endif

#include <fstream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class SOA3D;);

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
class SOA3D : Container3D<SOA3D<T>> {

private:
  bool fAllocated = false;
  size_t fSize = 0, fCapacity = 0;
  T *fX = nullptr, *fY = nullptr, *fZ = nullptr;

public:
  typedef T value_type;

  VECCORE_ATT_HOST_DEVICE
  SOA3D(T *x, T *y, T *z, size_t size);

  VECCORE_ATT_HOST_DEVICE
  SOA3D(size_t size);

  /// @brief Construct aligned data using specialized allocator.
  /// @details To allocate the full object, call this constructor with placement new.
  /// @param size Size of the internal arrays
  /// @param a Aligned allocator, pre-initialized to fit the content
  VECCORE_ATT_HOST_DEVICE
  SOA3D(size_t size, AlignedAllocator &a);

  SOA3D(SOA3D<T> const &other);

  SOA3D() = default;

  VECCORE_ATT_HOST_DEVICE
  SOA3D &operator=(SOA3D<T> const &other);

  VECCORE_ATT_HOST_DEVICE
  ~SOA3D();

  /// @brief Compute size of a buffer to hold the aligned arrays fX, fY and fZ
  /// @param initSize Number of elements
  /// @return Size to allocate
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static size_t aligned_sizeof_data(size_t initSize);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t size() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t capacity() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void resize(size_t newSize);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void reserve(size_t newCapacity);

  VECGEOM_FORCE_INLINE
  void clear();

  // Element access methods

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<T> operator[](size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T x(size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &x(size_t index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T *x();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const *x() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T y(size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &y(size_t index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T *y();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const *y() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T z(size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &z(size_t index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T *z();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const *z() const;

  // Element manipulation methods

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void set(size_t index, T x, T y, T z);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void set(size_t index, Vector3D<T> const &vec);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void push_back(T x, T y, T z);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void push_back(Vector3D<T> const &vec);

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::SOA3D<T>> CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu) const;
  DevicePtr<cuda::SOA3D<T>> CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu, size_t size) const;
#endif // VECGEOM_CUDA_INTERFACE

  void ToFile(std::string /*filename*/) const;
  int FromFile(std::string /*filename*/);

private:
  VECCORE_ATT_HOST_DEVICE
  void Allocate();

  VECCORE_ATT_HOST_DEVICE
  void Deallocate();
};

#if defined(GCC_DIAG_POP_NEEDED)

#pragma GCC diagnostic pop
#undef GCC_DIAG_POP_NEEDED

#endif

template <typename T>
VECCORE_ATT_HOST_DEVICE SOA3D<T>::SOA3D(T *xval, T *yval, T *zval, size_t sz)
    : fAllocated(false), fSize(sz), fCapacity(fSize), fX(xval), fY(yval), fZ(zval)
{
}

template <typename T>
VECCORE_ATT_HOST_DEVICE SOA3D<T>::SOA3D(size_t initSize, AlignedAllocator &a)
    : fAllocated(false), fSize(initSize), fCapacity(initSize)
{
  fX = a.aligned_alloc<T>(initSize, kAlignmentBoundary);
  fY = a.aligned_alloc<T>(initSize, kAlignmentBoundary);
  fZ = a.aligned_alloc<T>(initSize, kAlignmentBoundary);
}

template <typename T>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE size_t SOA3D<T>::aligned_sizeof_data(size_t initSize)
{
  // Size for separate aligned arrays fX, fY and fZ
  return 3 * AlignedAllocator::aligned_sizeof<T>(initSize, kAlignmentBoundary);
}

template <typename T>
VECCORE_ATT_HOST_DEVICE SOA3D<T>::SOA3D(size_t sz) : fSize(sz), fCapacity(sz)
{
  Allocate();
}

template <typename T>
SOA3D<T>::SOA3D(SOA3D<T> const &rhs) : fAllocated(false), fSize(rhs.fSize), fCapacity(rhs.fCapacity)
{
  if (rhs.fAllocated) {
    Allocate();
    copy(rhs.fX, rhs.fX + rhs.fSize, fX);
    copy(rhs.fY, rhs.fY + rhs.fSize, fY);
    copy(rhs.fZ, rhs.fZ + rhs.fSize, fZ);
  } else {
    fX = rhs.fX;
    fY = rhs.fY;
    fZ = rhs.fZ;
  }
}

template <typename T>
VECCORE_ATT_HOST_DEVICE SOA3D<T> &SOA3D<T>::operator=(SOA3D<T> const &rhs)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  fSize     = rhs.fSize;
  fCapacity = rhs.fCapacity;
  Deallocate();
  if (rhs.fAllocated && rhs.fSize > 0) {
    Allocate();
    copy(rhs.fX, rhs.fX + rhs.fSize, fX);
    copy(rhs.fY, rhs.fY + rhs.fSize, fY);
    copy(rhs.fZ, rhs.fZ + rhs.fSize, fZ);
  } else {
    fX = rhs.fX;
    fY = rhs.fY;
    fZ = rhs.fZ;
  }
#else
  fAllocated = false;
  fSize      = rhs.fSize;
  fCapacity  = rhs.fCapacity;
  fX         = rhs.fX;
  fY         = rhs.fY;
  fZ         = rhs.fZ;
#endif
  return *this;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE SOA3D<T>::~SOA3D()
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  Deallocate();
#endif
}

template <typename T>
VECCORE_ATT_HOST_DEVICE size_t SOA3D<T>::size() const
{
  return fSize;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE size_t SOA3D<T>::capacity() const
{
  return fCapacity;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void SOA3D<T>::resize(size_t newSize)
{
  assert(newSize <= fCapacity);
  fSize = newSize;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void SOA3D<T>::reserve(size_t newCapacity)
{
  fCapacity = newCapacity;
  T *xNew, *yNew, *zNew;
  xNew  = AlignedAllocate<T>(fCapacity);
  yNew  = AlignedAllocate<T>(fCapacity);
  zNew  = AlignedAllocate<T>(fCapacity);
  fSize = (fSize > fCapacity) ? fCapacity : fSize;
  if (fX && fY && fZ) {
    copy(fX, fX + fSize, xNew);
    copy(fY, fY + fSize, yNew);
    copy(fZ, fZ + fSize, zNew);
  }
  Deallocate();
  fX         = xNew;
  fY         = yNew;
  fZ         = zNew;
  fAllocated = true;
}

template <typename T>
void SOA3D<T>::clear()
{
  Deallocate();
  fSize     = 0;
  fCapacity = 0;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void SOA3D<T>::Allocate()
{
  if (fCapacity == 0) return;

  fX         = AlignedAllocate<T>(fCapacity);
  fY         = AlignedAllocate<T>(fCapacity);
  fZ         = AlignedAllocate<T>(fCapacity);
  fAllocated = true;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void SOA3D<T>::Deallocate()
{
  if (fAllocated) {
    AlignedFree(fX);
    AlignedFree(fY);
    AlignedFree(fZ);
  }
  fAllocated = false;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE Vector3D<T> SOA3D<T>::operator[](size_t index) const
{
  return Vector3D<T>(fX[index], fY[index], fZ[index]);
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T SOA3D<T>::x(size_t index) const
{
  return fX[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T &SOA3D<T>::x(size_t index)
{
  return fX[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T *SOA3D<T>::x()
{
  return fX;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T const *SOA3D<T>::x() const
{
  return fX;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T SOA3D<T>::y(size_t index) const
{
  return fY[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T &SOA3D<T>::y(size_t index)
{
  return fY[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T *SOA3D<T>::y()
{
  return fY;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T const *SOA3D<T>::y() const
{
  return fY;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T SOA3D<T>::z(size_t index) const
{
  return fZ[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T &SOA3D<T>::z(size_t index)
{
  return fZ[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T *SOA3D<T>::z()
{
  return fZ;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T const *SOA3D<T>::z() const
{
  return fZ;
}

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void SOA3D<T>::set(size_t index, T xval, T yval, T zval)
{
// not asserting in case of NVCC -- still getting annoying
// errors on CUDA < 8.0
#ifndef VECCORE_CUDA
  assert(index < fCapacity);
#endif
  fX[index] = xval;
  fY[index] = yval;
  fZ[index] = zval;
}

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void SOA3D<T>::set(size_t index, Vector3D<T> const &vec)
{
// not asserting in case of NVCC -- still getting annoying
// errors on CUDA < 8.0
#ifndef VECCORE_CUDA
  assert(index < fCapacity);
#endif
  fX[index] = vec[0];
  fY[index] = vec[1];
  fZ[index] = vec[2];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void SOA3D<T>::push_back(T xval, T yval, T zval)
{
  fX[fSize] = xval;
  fY[fSize] = yval;
  fZ[fSize] = zval;
  ++fSize;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void SOA3D<T>::push_back(Vector3D<T> const &vec)
{
  push_back(vec[0], vec[1], vec[2]);
}

template <typename T>
void SOA3D<T>::ToFile(std::string filename) const
{
  std::ofstream outfile(filename, std::ios::binary);
  outfile.write(reinterpret_cast<const char *>(&fSize), sizeof(fSize));
  outfile.write(reinterpret_cast<const char *>(&fCapacity), sizeof(fCapacity));
  outfile.write(reinterpret_cast<char *>(fX), fCapacity * sizeof(T));
  outfile.write(reinterpret_cast<char *>(fY), fCapacity * sizeof(T));
  outfile.write(reinterpret_cast<char *>(fZ), fCapacity * sizeof(T));
}

// read SOA3D from file serialized with ToFile
// returns number of elements read or -1 if failure
template <typename T>
int SOA3D<T>::FromFile(std::string filename)
{
  // should be called on an already allocated object
  decltype(fSize) s;
  decltype(fCapacity) cap;
  std::ifstream fin(filename, std::ios::binary);
  fin.read(reinterpret_cast<char *>(&s), sizeof(s));
  if (!fin) return -1;
  fin.read(reinterpret_cast<char *>(&cap), sizeof(cap));
  if (!fin) return -2;
  //  if (cap != fCapacity || s != fSize)
  //    std::cerr << " warning: reading from SOA3D with different size\n";

  fin.read(reinterpret_cast<char *>(fX), fCapacity * sizeof(T));
  if (!fin) return -3;

  fin.read(reinterpret_cast<char *>(fY), fCapacity * sizeof(T));
  if (!fin) return -4;

  fin.read(reinterpret_cast<char *>(fZ), fCapacity * sizeof(T));
  if (!fin) return -5;

  return fCapacity;
}

#ifdef VECGEOM_CUDA_INTERFACE

template <typename T>
DevicePtr<cuda::SOA3D<T>> SOA3D<T>::CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu) const
{
  xGpu.ToDevice(fX, fSize);
  yGpu.ToDevice(fY, fSize);
  zGpu.ToDevice(fZ, fSize);

  DevicePtr<cuda::SOA3D<T>> gpu_ptr;
  gpu_ptr.Allocate();
  gpu_ptr.Construct(xGpu, yGpu, zGpu, fSize);
}

template <typename T>
DevicePtr<cuda::SOA3D<T>> SOA3D<T>::CopyToGpu(DevicePtr<T> xGpu, DevicePtr<T> yGpu, DevicePtr<T> zGpu,
                                              size_t count) const
{
  xGpu.ToDevice(fX, count);
  yGpu.ToDevice(fY, count);
  zGpu.ToDevice(fZ, count);

  DevicePtr<cuda::SOA3D<T>> gpu_ptr;
  gpu_ptr.Allocate();
  gpu_ptr.Construct(xGpu, yGpu, zGpu, fSize);
}

#endif // VECGEOM_CUDA_INTERFACE
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_SOA3D_H_
