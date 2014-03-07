#include "base/aos3d.h"
#include "base/soa3d.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda_backend.cuh"
#endif

namespace vecgeom {

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
SOA3D<Type>::SOA3D(TrackContainer<Type> const &other) {
  SOA3D(other.size());
  const unsigned count = other.size();
  for (int i = 0; i < count; ++i) Set(i, other[i]);
}

#ifdef VECGEOM_CUDA

template <typename Type>
SOA3D<Type> SOA3D<Type>::CopyToGpu() const {
  const int count = this->size();
  const int mem_size = count*sizeof(Type);
  Type *x, *y, *z;
  cudaMalloc(static_cast<void**>(&x), mem_size);
  cudaMalloc(static_cast<void**>(&y), mem_size);
  cudaMalloc(static_cast<void**>(&z), mem_size);
  cudaMemcpy(x, x_, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(y, y_, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(z, z_, mem_size, cudaMemcpyHostToDevice);
  return SOA3D<Type>(x, y, z, count);
}

/**
 * Only works for SOA pointing to the GPU.
 */
template <typename Type>
void SOA3D<Type>::FreeFromGpu() {
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(z_);
  CudaAssertError();
}

#endif // VECGEOM_CUDA

} // End namespace vecgeom