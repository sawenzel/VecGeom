#include "base/aos3d.h"
#include "base/vector3d.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename Type>
__global__
void AOS3DCopyToGpuInterfaceKernel(
    vecgeom_cuda::Vector3D<Type> *const data, const unsigned size,
    vecgeom_cuda::AOS3D<Type> *const placement) {
  new(placement) vecgeom_cuda::AOS3D<Type>(data, size);
}

template <typename Type>
AOS3D<Type>* AOS3DCopyToGpuInterfaceTemplate(Vector3D<Type> *const data,
                                             const unsigned size) {
  vecgeom_cuda::AOS3D<Type> *const aos3d_gpu =
      AllocateOnGpu<vecgeom_cuda::AOS3D<Type> >();
  AOS3DCopyToGpuInterfaceKernel<<<1, 1>>>(
    reinterpret_cast<vecgeom_cuda::Vector3D<Type> *>(data),
    size,
    aos3d_gpu
  );
  return reinterpret_cast<AOS3D<Type> *>(aos3d_gpu);
}

AOS3D<Precision>* AOS3DCopyToGpuInterface(Vector3D<Precision> *const data,
                                          const unsigned size) {
  return AOS3DCopyToGpuInterfaceTemplate(data, size);
}

#endif

} // End namespace vecgeom