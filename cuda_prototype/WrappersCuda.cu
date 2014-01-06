#include "LibraryCuda.cuh"
#include "Kernel.h"
#include "Box.h"

namespace kernel {
namespace box {

__global__
void ContainsWrapper(const Vector3D<CudaFloat> box_pos,
                     const Vector3D<CudaFloat> box_dim,
                     SOA3D<double> const points, bool *output) {

  const int index = ThreadIndex();
  if (index >= points.size()) return;
  output[index] = Contains<kCuda>(box_pos, box_dim,
                                  points[index].ToFloatDevice());
}

} // End namespace box
} // End namespace kernel

template <>
void Box::Contains<kCuda>(SOA3D<double> const &points, bool *output) const {

  const int blocks_per_grid = BlocksPerGrid(points.size());
  kernel::box::ContainsWrapper<<<threads_per_block, blocks_per_grid>>>(
    trans_matrix->Translation().ToFloatHost(),
    dimensions.ToFloatHost(),
    points,
    output
  );

}