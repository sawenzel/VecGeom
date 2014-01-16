#include "LibraryCuda.cuh"
#include "Kernel.h"
#include "Box.h"

namespace kernel {
namespace box {

__global__
void ContainsWrapper(const Vector3D<CudaFloat> dimensions,
                     TransMatrix<CudaFloat> const * const trans_matrix,
                     SOA3D<double> const points, bool *output) {

  const int index = ThreadIndex();
  if (index >= points.size()) return;
  Vector3D<double> point = points[index];
  output[index] = Contains<kCuda>(dimensions, trans_matrix,
                                  DeviceVector(point));
}

__global__
void DistanceToInWrapper(const Vector3D<CudaFloat> dimensions,
                         TransMatrix<CudaFloat> const * const trans_matrix,
                         SOA3D<double> const pos, SOA3D<double> const dir,
                         double const *step_max, double *distance) {

  const int index = ThreadIndex();
  if (index >= pos.size()) return;
  const CudaFloat dist = DistanceToIn<kCuda>(dimensions, trans_matrix,
                                              DeviceVector(pos[index]),
                                              DeviceVector(dir[index]),
                                              CudaFloat(step_max[index]));
  distance[index] = double(dist);
}

} // End namespace box
} // End namespace kernel

void Box::Contains(SOA3D<double> const &points,
                   bool *output) const {

  const LaunchParameters launch(points.size());
  kernel::box::ContainsWrapper<<<launch.grid_size, launch.block_size>>>(
    DeviceVector(dimensions),
    trans_matrix_cuda,
    points,
    output
  );

}

void Box::DistanceToIn(SOA3D<double> const &pos,
                       SOA3D<double> const &dir,
                       double const *step_max,
                       double *distance) const {

  const LaunchParameters launch(pos.size());
  kernel::box::DistanceToInWrapper<<<launch.grid_size, launch.block_size>>>(
    DeviceVector(dimensions),
    trans_matrix_cuda,
    pos,
    dir,
    step_max,
    distance
  );
  CheckCudaError();
}