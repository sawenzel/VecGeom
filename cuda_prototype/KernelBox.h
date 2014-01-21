#ifndef KERNELBOX_H
#define KERNELBOX_H

#include "LibraryGeneric.h"

namespace kernel {

namespace box {

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
void Contains(
    Vector3D   <typename Impl<it>::float_t> const &dimensions,
    TransMatrix<typename Impl<it>::float_t> const * const matrix,
    Vector3D   <typename Impl<it>::float_v> const &point,
    typename Impl<it>::bool_v               &output) {

  const Vector3D<typename Impl<it>::float_v> local =
      matrix->Transform(point);

  typename Impl<it>::bool_v inside[3];
  for (int i = 0; i < 3; ++i) {
    inside[i] = Abs<it>(local[i]) < dimensions[i];
    if (Impl<it>::early_return) {
      if (!inside[i]) {
        output = Impl<it>::kFalse;
        return;
      }
    }
  }

  if (Impl<it>::early_return) {
    output = Impl<it>::kTrue;
  } else {
    output = inside[0] && inside[1] && inside[2];
  }
}

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
void DistanceToIn(
    Vector3D   <typename Impl<it>::float_t> const &dimensions,
    TransMatrix<typename Impl<it>::float_t> const * const matrix,
    Vector3D   <typename Impl<it>::float_v> const &pos,
    Vector3D   <typename Impl<it>::float_v> const &dir,
    typename Impl<it>::float_v              const &step_max,
    typename Impl<it>::float_v              &distance) {

  typedef typename Impl<it>::float_v Float;
  typedef typename Impl<it>::bool_v Bool;

  Vector3D<Float> safety;
  Vector3D<Float> pos_local;
  Vector3D<Float> dir_local;
  Bool hit = Impl<it>::kFalse;
  Bool done = Impl<it>::kFalse;
  distance = kInfinity;

  matrix->Transform(pos, pos_local);
  matrix->TransformRotation(dir, dir_local);

  safety[0] = Abs<it>(pos_local[0]) - dimensions[0];
  safety[1] = Abs<it>(pos_local[1]) - dimensions[1];
  safety[2] = Abs<it>(pos_local[2]) - dimensions[2];

  done |= (safety[0] >= step_max ||
           safety[1] >= step_max ||
           safety[2] >= step_max);
  if (done == Impl<it>::kTrue) return;

  Float next, coord1, coord2;

  // x
  next = safety[0] / Abs<it>(dir_local[0] + kTiny);
  coord1 = pos_local[1] + next * dir_local[1];
  coord2 = pos_local[2] + next * dir_local[2];
  hit = safety[0] > 0 &&
        pos_local[0] * dir_local[0] < 0 &&
        Abs<it>(coord1) <= dimensions[1] &&
        Abs<it>(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, distance);
  done |= hit;
  if (done == Impl<it>::kTrue) return;

  // y
  next = safety[1] / Abs<it>(dir_local[1] + kTiny);
  coord1 = pos_local[0] + next * dir_local[0];
  coord2 = pos_local[2] + next * dir_local[2];
  hit = safety[1] > 0 &&
        pos_local[1] * dir_local[1] < 0 &&
        Abs<it>(coord1) <= dimensions[0] &&
        Abs<it>(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, distance);
  done |= hit;
  if (done == Impl<it>::kTrue) return;

  // z
  next = safety[2] / Abs<it>(dir_local[2] + kTiny);
  coord1 = pos_local[0] + next * dir_local[0];
  coord2 = pos_local[1] + next * dir_local[1];
  hit = safety[2] > 0 &&
        pos_local[2] * dir_local[2] < 0 &&
        Abs<it>(coord1) <= dimensions[0] &&
        Abs<it>(coord2) <= dimensions[1];
  MaskedAssign(!done && hit, next, distance);

}

} // End namespace box

} // End namespace kernel

#endif /* KERNELBOX_H */