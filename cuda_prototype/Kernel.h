#ifndef KERNEL_H
#define KERNEL_H

#include "LibraryGeneric.h"

namespace kernel {

template <Ct ct>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
typename CtTraits<ct>::bool_v Contains(
    Vector3D<typename CtTraits<ct>::float_t> const &box_pos,
    Vector3D<typename CtTraits<ct>::float_t> const &box_dim,
    Vector3D<typename CtTraits<ct>::float_v> const &point) {
  typename CtTraits<ct>::bool_v inside[3];
  for (int i = 0; i < 3; ++i) {
    inside[i] = abs(point[i] - box_pos[i]) < box_dim[i];
  }
  return inside[0] && inside[1] && inside[2];
}

} // End namespace kernel

#endif /* KERNEL_H */