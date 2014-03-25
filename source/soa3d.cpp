#include "base/soa3d.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename Type>
__global__
void SOA3DCopyToGpuInterfaceKernel(
    Type *const x, Type *const y, Type *const z, const unsigned size,
    vecgeom_cuda::SOA3D<Type> *const placement) {
  new(placement) vecgeom_cuda::SOA3D<Type>(x, y, z, size);
}

template <typename Type>
SOA3D<Type>* SOA3DCopyToGpuInterfaceTemplate(Type *const x, Type *const y,
                                             Type *const z,
                                             const unsigned size) {
  vecgeom_cuda::SOA3D<Type> *const soa3d_gpu =
      AllocateOnGpu<vecgeom_cuda::SOA3D<Type> >();
  SOA3DCopyToGpuInterfaceKernel<<<1, 1>>>(x, y, z, size, soa3d_gpu);
  return reinterpret_cast<SOA3D<Type> *>(soa3d_gpu);
}

SOA3D<Precision>* SOA3DCopyToGpuInterface(Precision *const x,
                                          Precision *const y,
                                          Precision *const z,
                                          const unsigned size) {
  return SOA3DCopyToGpuInterfaceTemplate(x, y, z, size);
}

#endif

} // End namespace vecgeom