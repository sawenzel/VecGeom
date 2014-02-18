#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/transformation_matrix.h"
#include "backend/cuda_backend.cuh"
#include "volumes/kernel/box_kernel.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"

using namespace vecgeom;

__device__
void compile_cuda() {
  CudaPrecision scalar;
  Vector3D<double> scalar_v;
  Vector3D<CudaPrecision> vector_v;
  SOA3D<CudaPrecision> soa;
  TransformationMatrix matrix;
  CudaBool output_inside;
  CudaPrecision output_distance;
  BoxInside<translation::kOrigin, rotation::kIdentity, kCuda>(
    scalar_v, matrix, vector_v, &output_inside
  );
  BoxDistanceToIn<translation::kOrigin, rotation::kIdentity, kCuda>(
    scalar_v, matrix, vector_v, vector_v, scalar, &output_distance
  );
}