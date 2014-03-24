#include "base/container.h"
#include "base/vector.h"

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename Type>
__global__
void ConstructOnGpu(Type *const arr, const int size,
                    vecgeom::Vector<Type> *gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::Vector<Type>(arr, size);
}

void ContainerGpuInterface(Precision *const arr, const int size,
                           vecgeom::Vector<Precision> *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(arr, size, gpu_ptr);
}

void ContainerGpuInterface(
    VPlacedVolume const **const arr, const int size,
    vecgeom::Vector<VPlacedVolume const*> *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(arr, size, gpu_ptr);
}

#endif

} // End namespace vecgeom