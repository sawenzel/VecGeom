#ifndef LAUNCHERCUDA_H
#define LAUNCHERCUDA_H

#include "LibraryCuda.cuh"

class LauncherCuda {

private:

  static const int threads_per_block;

public:

  inline __attribute__((always_inline))
  static int BlocksPerGrid(const int threads) {
    return (threads - 1) / threads_per_block + 1;
  }

  static void Contains(Vector3D<CudaFloat> const& /*box_pos*/,
                       Vector3D<CudaFloat> const& /*box_dim*/,
                       SOA3D_CUDA_Float const& /*points*/,
                       CudaBool* /*output*/);

private:

  LauncherCuda() {}

  template <typename Type>
  inline __attribute__((always_inline))
  static Type* AllocateOnGPU(const int count) {
    Type *ptr;
    cudaMalloc((void**)&ptr, count*sizeof(Type));
    return ptr;
  }

  template <typename Type>
  inline __attribute__((always_inline))
  static void CopyFromGPU(Type *const src, Type *const tgt, const int count) {
    cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyDeviceToHost);
  }

};

#endif /* LAUNCHERCUDA_H */