#ifndef KERNEL_H
#define KERNEL_H

#include "LibraryGeneric.h"

namespace kernel {

namespace box {

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
typename ImplTraits<it>::bool_v Contains(
    Vector3D<typename ImplTraits<it>::float_t> const &box_pos,
    Vector3D<typename ImplTraits<it>::float_t> const &box_dim,
    Vector3D<typename ImplTraits<it>::float_v> const &point) {

  typename ImplTraits<it>::bool_v inside[3];
  for (int i = 0; i < 3; ++i) {
    inside[i] = abs(point[i] - box_pos[i]) < box_dim[i];
  }
  return inside[0] && inside[1] && inside[2];
}

// template <Ct ct>
// inline __attribute__((always_inline))
// CUDA_HEADER_BOTH
// Vector3D<typename ImplTraits<it>::float_t> DistanceToIn(
//     Vector3D<typename ImplTraits<it>::float_t> const &box_pos,
//     Vector3D<typename ImplTraits<it>::float_t> const &box_dim,
//     Vector3D<typename ImplTraits<it>::float_t> const &point,
//     Vector3D<typename ImplTraits<it>::float_t> const &dir, const double step) {

//   typedef typename CtTraits<it>::float_t Float;

//   const Float delta = 1e-9;

//   Vector3D<Float> point_trans;
//   matrix->MasterToLocal<tid,rid>(x, aPoint);

// }

} // End namespace box

} // End namespace kernel

#endif /* KERNEL_H */