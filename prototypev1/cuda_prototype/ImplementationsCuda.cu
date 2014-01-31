#include "LibraryCuda.cuh"
#include "KernelBox.h"
#include "Box.h"
#include "Tube.h"

namespace kernel {
namespace box {

__global__
void ContainsWrapper(const Vector3D<CudaFloat> dimensions,
                     TransMatrix<CudaFloat> const * const trans_matrix,
                     SOA3D<double> const points, bool *output) {

  const int index = ThreadIndex();
  if (index >= points.size()) return;
  Vector3D<double> point = points[index];
  Contains<kCuda>(dimensions, trans_matrix, DeviceVector(point), output[index]);
}

__global__
void DistanceToInWrapper(const Vector3D<CudaFloat> dimensions,
                         TransMatrix<CudaFloat> const * const trans_matrix,
                         SOA3D<double> const pos, SOA3D<double> const dir,
                         double const *step_max, double *distance) {

  const int index = ThreadIndex();
  if (index >= pos.size()) return;
  CudaFloat dist;
  DistanceToIn<kCuda>(dimensions, trans_matrix, DeviceVector(pos[index]),
                      DeviceVector(dir[index]), CudaFloat(step_max[index]),
                      dist);
  distance[index] = double(dist);
}

} // End namespace box
} // End namespace kernel

void Box::Contains(SOA3D<double> const &points,
                   bool *output) const {

  const LaunchParameters launch(points.size());
  kernel::box::ContainsWrapper<<<launch.grid_size, launch.block_size>>>(
    DeviceVector(parameters->dimensions),
    trans_matrix,
    points,
    output
  );
  CheckCudaError();
}

void Box::DistanceToIn(SOA3D<double> const &pos,
                       SOA3D<double> const &dir,
                       double const *step_max,
                       double *distance) const {

  const LaunchParameters launch(pos.size());
  kernel::box::DistanceToInWrapper<<<launch.grid_size, launch.block_size>>>(
    DeviceVector(parameters->dimensions),
    trans_matrix,
    pos,
    dir,
    step_max,
    distance
  );
  CheckCudaError();
}

void Tube::Contains(SOA3D<double> const &points,
                    bool *output) const {
  // NYI
}

void Tube::DistanceToIn(SOA3D<double> const &pos,
                        SOA3D<double> const &dir,
                        double const *steps_max,
                        double *distance) const {
  // NYI
}