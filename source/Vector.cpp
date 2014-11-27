/// \file Vector.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/Vector.h"

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename Type>
__global__
void ConstructOnGpu(Type *const arr, const int size,
                    void *gpu_ptr) {
  new(gpu_ptr) vecgeom::cuda::Vector<Type>(arr, size);
}

void Vector_CopyToGpu(Precision *const arr, const int size,
                      void *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(arr, size, gpu_ptr);
}

void Vector_CopyToGpu(
    VPlacedVolume const **const arr, const int size,
    void *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(arr, size, gpu_ptr);
}

#endif

} // End namespace vecgeom
