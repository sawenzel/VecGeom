/// \file SOA3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/SOA3D.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

template <typename T>
__global__
void ConstructOnGpu(T *x, T *y, T *z, size_t size,
                    vecgeom_cuda::SOA3D<T> *placement) {
  new(placement) vecgeom_cuda::SOA3D<T>(x, y, z, size);
}

template <typename T>
SOA3D<T>* SOA3D_CopyToGpuTemplate(T *x, T *y, T *z, size_t size) {
  vecgeom_cuda::SOA3D<T> *soa3DGpu =
      AllocateOnGpu<vecgeom_cuda::SOA3D<T> >();
  ConstructOnGpu<<<1, 1>>>(x, y, z, size, soa3DGpu);
  return reinterpret_cast<SOA3D<T> *>(soa3DGpu);
}

SOA3D<Precision>* SOA3D_CopyToGpu(Precision *x, Precision *y, Precision *z,
                                  size_t size) {
  return SOA3D_CopyToGpuTemplate(x, y, z, size);
}

#endif

} // End namespace vecgeom