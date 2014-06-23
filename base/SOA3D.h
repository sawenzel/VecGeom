/// \file SOA3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_SOA3D_H_
#define VECGEOM_BASE_SOA3D_H_

#include "base/Global.h"

#include "base/TrackContainer.h"
#include "base/Vector3D.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom_cuda { template <typename Type> class SOA3D; }

namespace VECGEOM_NAMESPACE {

/**
 * @brief Stores arrays of three coordinates in a structure of arrays fashion.
 */
template <typename Type>
class SOA3D : public TrackContainer<Type> {

private:

  Type *x_, *y_, *z_;

public:

  /**
   * Initializes the SOA with existing data arrays, performing no allocation.
   */
  VECGEOM_CUDA_HEADER_BOTH
    SOA3D(Type *const /* x */ , Type *const /* y */, Type *const /* z */, const int size);

  /**
   * Initializes the SOA with a fixed size, allocating an aligned array for each
   * coordinate of the specified size.
   */
  SOA3D(const int size);

  /**
   * Performs a deep copy from another SOA.
   */
  VECGEOM_CUDA_HEADER_BOTH
  SOA3D(TrackContainer<Type> const &other);

  // /**
  //  * Copy constructor
  //  */
  // VECGEOM_CUDA_HEADER_BOTH
  // SOA3D(SOA3D const &other);

  /**
   * assignment operator
   */
  VECGEOM_CUDA_HEADER_BOTH
  SOA3D* operator=(SOA3D const &other);

  ~SOA3D();

  /**
   * Constructs a vector across all three coordinates from the given index. 
   */ 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(x_[index], y_[index], z_[index]);
  }

  // Element access methods.
  // Can be used to manipulate content if necessary.

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& x(const int index) const { return x_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& x(const int index) { return x_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* x() { return x_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& y(const int index) const { return y_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& y(const int index) { return y_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* y() { return y_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& z(const int index) const { return z_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& z(const int index) { return z_[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type* z() { return z_; }

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
  /**
   * Allocates and copies the data of this SOA to the GPU, then creates and
   * returns a new SOA object that points to GPU memory.
   */
  SOA3D<Type>* CopyToGpu(Type *const x_gpu, Type *const y_gpu,
                         Type *const z_gpu) const;
  SOA3D<Type>* CopyToGpu(Type *const x_gpu, Type *const y_gpu,
                         Type *const z_gpu, const int size) const;
  #endif // VECGEOM_CUDA

private:

  VECGEOM_INLINE
  void Allocate();

};

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<Type>::SOA3D(Type *const xarray, Type *const yarray, Type *const zarray,
                   const int memory_size)
    : TrackContainer<Type>(memory_size, false), x_(xarray), y_(yarray), z_(zarray) {
  this->size_ = memory_size;
}

template <typename Type>
  SOA3D<Type>::SOA3D(const int size) : TrackContainer<Type>(size, true), x_(NULL), y_(NULL),
    z_(NULL)
 {
  Allocate();
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<Type>::SOA3D(TrackContainer<Type> const &other)
  : TrackContainer<Type>(other.size_, true),  x_(NULL), y_(NULL),
    z_(NULL) {
  Allocate();
  this->size_ = other.size_;
  for (int i = 0, i_end = this->size_; i < i_end; ++i) Set(i, other[i]);
}

template <typename Type>
SOA3D<Type>::~SOA3D() {
  if (this->allocated_) {
    _mm_free(x_);
    _mm_free(y_);
    _mm_free(z_);
  }
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SOA3D<Type>::Set(const int index, const Type xcomp, const Type ycomp,
                      const Type zcomp) {
  // assert(index < this->memory_size_);
  if (index >= this->size_) this->size_ = index+1;
  x_[index] = xcomp;
  y_[index] = ycomp;
  z_[index] = zcomp;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SOA3D<Type>::Set(const int index, Vector3D<Type> const &vec) {
  Set(index, vec[0], vec[1], vec[2]);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SOA3D<Type>::push_back(const Type xcomp, const Type ycomp, const Type zcomp) {
  // assert(this->size_ < this->memory_size_);
  x_[this->size_] = xcomp;
  y_[this->size_] = ycomp;
  z_[this->size_] = zcomp;
  this->size_++;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SOA3D<Type>::push_back(Vector3D<Type> const &vec) {
  push_back(vec[0], vec[1], vec[2]);
}

template <typename Type>
void SOA3D<Type>::Allocate() {
  x_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*this->memory_size_,
                          kAlignmentBoundary));
  y_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*this->memory_size_,
                          kAlignmentBoundary));
  z_ = static_cast<Type*>(_mm_malloc(sizeof(Type)*this->memory_size_,
                          kAlignmentBoundary));
}

#ifdef VECGEOM_CUDA

template <typename Type> class SOA3D;

SOA3D<Precision>* SOA3D_CopyToGpu(Precision *const x, Precision *const y,
                                  Precision *const z, const int size);

template <typename Type>
SOA3D<Type>* SOA3D<Type>::CopyToGpu(Type *const x_gpu,
                                    Type *const y_gpu,
                                    Type *const z_gpu) const {
  const int count = this->size_;
  const int mem_size = count*sizeof(Type);
  vecgeom::CopyToGpu(x_, x_gpu, mem_size);
  vecgeom::CopyToGpu(y_, y_gpu, mem_size);
  vecgeom::CopyToGpu(z_, z_gpu, mem_size);
  return SOA3D_CopyToGpu(x_gpu, y_gpu, z_gpu, count);
}

template <typename Type>
SOA3D<Type>* SOA3D<Type>::CopyToGpu(Type *const x_gpu,
                                    Type *const y_gpu,
                                    Type *const z_gpu,
                                    const int count) const {
  const int mem_size = count*sizeof(Type);
  vecgeom::CopyToGpu(x_, x_gpu, mem_size);
  vecgeom::CopyToGpu(y_, y_gpu, mem_size);
  vecgeom::CopyToGpu(z_, z_gpu, mem_size);
  return SOA3D_CopyToGpu(x_gpu, y_gpu, z_gpu, count);
}

#endif // VECGEOM_CUDA

} // End global namespace

#endif // VECGEOM_BASE_SOA3D_H_
