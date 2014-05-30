/// \file Container.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_CONTAINER_H_
#define VECGEOM_BASE_CONTAINER_H_

#include "base/Global.h"

#include "base/Array.h"
#include "base/Iterator.h"
#include "backend/Backend.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif
 

namespace VECGEOM_NAMESPACE {

/**
 * @brief Std-like container base class compatible with CUDA.
 * @details Derived classes implement random access indexing (time complexity
 *          can very), size and constant iterator to content.
 */
template <typename Type>
class Container {

public:

  VECGEOM_CUDA_HEADER_BOTH
  Container() {}

  virtual ~Container() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type& operator[](const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Type const& operator[](const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Iterator<Type> begin() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Iterator<Type> end() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual int size() const =0;

  #ifdef VECGEOM_CUDA_INTERFACE
  Vector<Type>* CopyToGpu(Type *const gpu_ptr_arr,
                          Vector<Type> *const gpu_ptr) const;
  #endif

};

} // End global namespace

namespace vecgeom {

template <typename Type> class Vector;
class VPlacedVolume;

void Container_CopyToGpu(Precision *const arr, const int size,
                         Vector<Precision> *const gpu_ptr);

void Container_CopyToGpu(VPlacedVolume const **const arr, const int size,
                         Vector<VPlacedVolume const*> *const gpu_ptr);

#ifdef VECGEOM_CUDA_INTERFACE
template <typename Type>
Vector<Type>* Container<Type>::CopyToGpu(Type *const gpu_ptr_arr,
                                         Vector<Type> *const gpu_ptr) const {
  Container_CopyToGpu(gpu_ptr_arr, this->size(), gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}
#endif // VECGEOM_CUDA_INTERFACE

} // End namespace vecgeom


#endif // VECGEOM_BASE_CONTAINER_H_