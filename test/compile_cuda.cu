#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/trans_matrix.h"
#include "backend/cuda_backend.cuh"
#include "volumes/kernel/box_kernel.h"

using namespace vecgeom;

__device__
void compile_cuda() {
  CudaDouble scalar;
  Vector3D<double> scalar_v;
  Vector3D<CudaDouble > vector_v;
  SOA3D<CudaDouble> soa;
  TransMatrix matrix;
  CudaBool output_inside;
  CudaDouble output_distance;
  BoxInside<kCuda>(scalar_v, matrix, vector_v, &output_inside);
  BoxDistanceToIn<kCuda>(scalar_v, matrix, vector_v, vector_v, scalar,
                         &output_distance);
}