/// \file AOS3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_AOS3D_H_
#define VECGEOM_BASE_AOS3D_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/base/Container3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/backend/scalar/Backend.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class AOS3D;);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
class AOS3D : Container3D<AOS3D<T>> {

private:
  bool fAllocated{false};
  size_t fSize{0};
  size_t fCapacity{0};
  Vector3D<T> *fContent{nullptr};

  typedef Vector3D<T> Vec_t;

public:
  typedef T value_type;

  VECCORE_ATT_HOST_DEVICE
  AOS3D(Vector3D<T> *data, size_t size);

  VECCORE_ATT_HOST_DEVICE
  AOS3D(size_t size);

  /// @brief Construct aligned data using specialized allocator.
  /// @details To allocate the full object, call this constructor with placement new.
  /// @param size Size of the internal arrays
  /// @param a Aligned allocator, pre-initialized to fit the content
  VECCORE_ATT_HOST_DEVICE
  AOS3D(size_t size, AlignedAllocator &a);

  AOS3D() = default;

  AOS3D(AOS3D<T> const &other);

  VECCORE_ATT_HOST_DEVICE
  AOS3D &operator=(AOS3D<T> const &other);

  VECCORE_ATT_HOST_DEVICE
  ~AOS3D();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t size() const;

  /// @brief Compute size of a buffer to hold the aligned vector content
  /// @param initSize Number of elements
  /// @return Size to allocate
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static size_t aligned_sizeof_data(size_t initSize);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t capacity() const;

  VECGEOM_FORCE_INLINE
  void resize(size_t newSize);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void reserve(size_t newCapacity);

  VECGEOM_FORCE_INLINE
  void clear();

  // Element access methods. Can be used to manipulate content.

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<T> operator[](size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<T> &operator[](size_t index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<T> *content();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<T> const *content() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T x(size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &x(size_t index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T y(size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &y(size_t index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T z(size_t index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &z(size_t index);

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
  DevicePtr<cuda::AOS3D<T>> CopyToGpu(DevicePtr<cuda::Vector3D<T>> contentGpu) const;
#endif

private:
  VECCORE_ATT_HOST_DEVICE
  void Deallocate();
};

template <typename T>
VECCORE_ATT_HOST_DEVICE AOS3D<T>::AOS3D(Vector3D<T> *in_content, size_t in_size)
    : fSize(in_size), fCapacity(fSize), fContent(in_content)
{
}

template <typename T>
VECCORE_ATT_HOST_DEVICE AOS3D<T>::AOS3D(size_t sz, AlignedAllocator &a) : fSize(sz), fCapacity(sz)
{
  fContent = a.aligned_alloc<Vector3D<T>>(sz, kAlignmentBoundary);
}

template <typename T>
VECCORE_ATT_HOST_DEVICE AOS3D<T>::AOS3D(size_t sz) : fSize(sz), fCapacity(sz)
{
  if (fCapacity > 0) reserve(fCapacity);
}

template <typename T>
AOS3D<T>::AOS3D(AOS3D<T> const &rhs) : fSize(rhs.fSize), fCapacity(rhs.fCapacity)
{
  *this = rhs;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE AOS3D<T> &AOS3D<T>::operator=(AOS3D<T> const &rhs)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  clear();
  if (rhs.fAllocated) {
    reserve(rhs.fCapacity);
    copy(rhs.fContent, rhs.fContent + rhs.fSize, fContent);
  } else {
    fContent   = rhs.fContent;
    fAllocated = false;
    fCapacity  = rhs.fCapacity;
  }
  fSize = rhs.fSize;
#else
  fAllocated = false;
  fSize      = rhs.fSize;
  fCapacity  = rhs.fCapacity;
  fContent   = rhs.fContent;
#endif
  return *this;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE AOS3D<T>::~AOS3D()
{
  Deallocate();
}

template <typename T>
VECCORE_ATT_HOST_DEVICE size_t AOS3D<T>::size() const
{
  return fSize;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE size_t AOS3D<T>::aligned_sizeof_data(size_t initSize)
{
  return initSize * sizeof(Vector3D<T>) + kAlignmentBoundary;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE size_t AOS3D<T>::capacity() const
{
  return fCapacity;
}

template <typename T>
void AOS3D<T>::resize(size_t newSize)
{
  assert(newSize <= fCapacity);
  fSize = newSize;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void AOS3D<T>::reserve(size_t newCapacity)
{
  fCapacity         = newCapacity;
  Vec_t *contentNew = fCapacity > 0 ? AlignedAllocate<Vec_t>(fCapacity) : nullptr;
  fSize             = (fSize > fCapacity) ? fCapacity : fSize;
  if (fContent && fSize > 0) {
    copy(fContent, fContent + fSize, contentNew);
  }
  Deallocate();
  fContent   = contentNew;
  fAllocated = fContent != nullptr;
}

template <typename T>
void AOS3D<T>::clear()
{
  Deallocate();
  fContent   = nullptr;
  fAllocated = false;
  fSize      = 0;
  fCapacity  = 0;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void AOS3D<T>::Deallocate()
{
  if (fAllocated) {
    AlignedFree(fContent);
  }
}

template <typename T>
VECCORE_ATT_HOST_DEVICE Vector3D<T> AOS3D<T>::operator[](size_t index) const
{
  return fContent[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE Vector3D<T> &AOS3D<T>::operator[](size_t index)
{
  return fContent[index];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE Vector3D<T> *AOS3D<T>::content()
{
  return fContent;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE Vector3D<T> const *AOS3D<T>::content() const
{
  return fContent;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T AOS3D<T>::x(size_t index) const
{
  return (fContent[index])[0];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T &AOS3D<T>::x(size_t index)
{
  return (fContent[index])[0];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T AOS3D<T>::y(size_t index) const
{
  return (fContent[index])[1];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T &AOS3D<T>::y(size_t index)
{
  return (fContent[index])[1];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T AOS3D<T>::z(size_t index) const
{
  return (fContent[index])[2];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE T &AOS3D<T>::z(size_t index)
{
  return (fContent[index])[2];
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void AOS3D<T>::set(size_t index, T in_x, T in_y, T in_z)
{
  (fContent[index])[0] = in_x;
  (fContent[index])[1] = in_y;
  (fContent[index])[2] = in_z;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void AOS3D<T>::set(size_t index, Vector3D<T> const &vec)
{
  fContent[index] = vec;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void AOS3D<T>::push_back(T in_x, T in_y, T in_z)
{
  (fContent[fSize])[0] = in_x;
  (fContent[fSize])[1] = in_y;
  (fContent[fSize])[2] = in_z;
  ++fSize;
}

template <typename T>
VECCORE_ATT_HOST_DEVICE void AOS3D<T>::push_back(Vector3D<T> const &vec)
{
  fContent[fSize] = vec;
  ++fSize;
}

#ifdef VECGEOM_CUDA_INTERFACE

template <typename T>
DevicePtr<cuda::AOS3D<T>> AOS3D<T>::CopyToGpu(DevicePtr<cuda::Vector3D<T>> contentGpu) const
{
  contentGpu.ToDevice(fContent, fSize);

  DevicePtr<cuda::AOS3D<T>> gpu_ptr;
  gpu_ptr.Allocate();
  gpu_ptr.Construct(contentGpu, fSize);
}

#endif // VECGEOM_CUDA_INTERFACE
} // namespace VECGEOM_IMPL_NAMESPACE
} // End namespace vecgeom

#endif // VECGEOM_BASE_AOS3D_H_
