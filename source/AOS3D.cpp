/// \file AOS3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/AOS3D.h"

#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename Type>
__global__
void ConstructOnGpu(
    vecgeom_cuda::Vector3D<Type> *const data, const unsigned size,
    vecgeom_cuda::AOS3D<Type> *const placement) {
  new(placement) vecgeom_cuda::AOS3D<Type>(data, size);
}

template <typename Type>
AOS3D<Type>* AOS3D_CopyToGpuTemplate(
    vecgeom_cuda::Vector3D<Type> *const data, const unsigned size) {
  vecgeom_cuda::AOS3D<Type> *const aos3d_gpu =
      AllocateOnGpu<vecgeom_cuda::AOS3D<Type> >();
  ConstructOnGpu<<<1, 1>>>(data, size, aos3d_gpu);
  return reinterpret_cast<AOS3D<Type> *>(aos3d_gpu);
}

AOS3D<Precision>* AOS3D_CopyToGpu(
    vecgeom_cuda::Vector3D<Precision> *const data, const unsigned size) {
  return AOS3D_CopyToGpuTemplate(data, size);
}

#endif

} // End namespace vecgeom