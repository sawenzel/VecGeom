/// \file AOS3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/AOS3D.h"

#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename T>
__global__
void ConstructOnGpu(
    vecgeom_cuda::Vector3D<T> *content, size_t size,
    vecgeom_cuda::AOS3D<T> *placement) {
  new(placement) vecgeom_cuda::AOS3D<T>(content, size);
}

template <typename T>
AOS3D<T>* AOS3D_CopyToGpuTemplate(
    vecgeom_cuda::Vector3D<T> *content, size_t size) {
  vecgeom_cuda::AOS3D<T> *aos3DGpu = AllocateOnGpu<vecgeom_cuda::AOS3D<T> >();
  ConstructOnGpu<<<1, 1>>>(content, size, aos3DGpu);
  return reinterpret_cast<AOS3D<T> *>(aos3DGpu);
}

AOS3D<Precision>* AOS3D_CopyToGpu(
    vecgeom_cuda::Vector3D<Precision> *content, size_t size) {
  return AOS3D_CopyToGpuTemplate(content, size);
}

#endif

} // End namespace vecgeom