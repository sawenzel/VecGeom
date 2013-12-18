#include "LauncherCuda.cuh"
#include "Kernel.h"

const int LauncherCuda::threads_per_block = 512;

__global__
void ContainsWrapper(Vector3D<CudaFloat> const box_pos,
                     Vector3D<CudaFloat> const box_dim,
                     SOA3D_CUDA_Float const points, CudaBool *output) {
  const int index = ThreadIndex();
  if (index >= points.size()) return;
  output[index] = kernel::Contains<kCuda>(box_pos, box_dim, points[index]);
}

void LauncherCuda::Contains(Vector3D<CudaFloat> const &box_pos,
                            Vector3D<CudaFloat> const &box_dim,
                            SOA3D_CUDA_Float const &points, CudaBool *output) {

  const int threads = points.size();
  const int blocks_per_grid = BlocksPerGrid(threads);

  SOA3D_CUDA_Float points_dev = points.CopyToGPU();
  CudaBool *output_dev = AllocateOnGPU<CudaBool>(threads);

  ContainsWrapper<<<threads_per_block, blocks_per_grid>>>(
    box_pos, box_dim, points_dev, output_dev
  );

  CopyFromGPU(output_dev, output, threads);
  points_dev.FreeFromGPU();
  cudaFree(output_dev);

}