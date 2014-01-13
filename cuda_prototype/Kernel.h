#ifndef KERNEL_H
#define KERNEL_H

#include "LibraryGeneric.h"

namespace kernel {
namespace box {

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
typename ImplTraits<it>::bool_v Contains(
    Vector3D<typename ImplTraits<it>::float_t> const &dimensions,
    TransMatrix<typename ImplTraits<it>::float_t> const * const matrix,
    Vector3D<typename ImplTraits<it>::float_v> const &point) {

  Vector3D<typename ImplTraits<it>::float_v> local = point;
  if (!matrix->IsIdentity()) {
    local = matrix->MasterToLocal(point);
  }

  typename ImplTraits<it>::bool_v inside[3];
  for (int i = 0; i < 3; ++i) {
    inside[i] = abs(local[i]) < dimensions[i];
    if (ImplTraits<it>::early_return) {
      if (!inside[i]) return ImplTraits<it>::kFalse;
    }
  }

  if (ImplTraits<it>::early_return) {
    return ImplTraits<it>::kTrue;
  } else {
    return inside[0] && inside[1] && inside[2];
  }
}

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
typename ImplTraits<it>::float_v DistanceToIn(
    Vector3D<typename ImplTraits<it>::float_t> const &dimensions,
    TransMatrix<typename ImplTraits<it>::float_t> const * const matrix,
    Vector3D<typename ImplTraits<it>::float_v> const &pos,
    Vector3D<typename ImplTraits<it>::float_v> const &dir,
    typename ImplTraits<it>::float_v const &step_max) {

  // Typedef templated types for readability
  typedef typename ImplTraits<it>::float_v Float;
  typedef typename ImplTraits<it>::bool_v Bool;
  typedef typename ImplTraits<it>::float_t ScalarFloat;

  const ScalarFloat kTiny(1e-20);
  Vector3D<Float> safety;
  Vector3D<Float> pos_local;
  Vector3D<Float> dir_local;
  Bool hit = ImplTraits<it>::kFalse;
  Bool done = ImplTraits<it>::kFalse;
  Float distance(kInfinity);

  matrix->MasterToLocal(pos, pos_local);
  matrix->MasterToLocal(dir, dir_local);

  safety[0] = abs(pos_local[0]) - dimensions[0];
  safety[1] = abs(pos_local[1]) - dimensions[1];
  safety[2] = abs(pos_local[2]) - dimensions[2];

  done |= (safety[0] >= step_max ||
           safety[1] >= step_max ||
           safety[2] >= step_max);
  if (done == ImplTraits<it>::kTrue) return distance;

  Vector3D<Float> next;
  Float coord1, coord2;

  // x
  next[0] = safety[0] / (abs(dir_local[0]) + kTiny);
  coord1 = pos_local[1] + next[0] * dir_local[1];
  coord2 = pos_local[2] + next[0] * dir_local[2];
  hit = safety[0] > 0 &&
        pos_local[0] * dir_local[0] < 0 &&
        (abs(coord1) <= dimensions[1] && abs(coord2) <= dimensions[2]);
  MaskedAssign(!done && hit, next[0], distance);
  done |= hit;
  if (done == ImplTraits<it>::kTrue) return distance;

  // y
  next[1] = safety[1] / (abs(dir_local[1]) + kTiny);
  coord1 = pos_local[0] + next[1] * dir_local[0];
  coord2 = pos_local[2] + next[1] * dir_local[2];
  hit = safety[1] > 0 &&
        pos_local[1] * dir_local[1] < 0 &&
        (abs(coord1) <= dimensions[0] && abs(coord2) <= dimensions[2]);
  MaskedAssign(!done && hit, next[1], distance);
  done |= hit;
  if (done == ImplTraits<it>::kTrue) return distance;

  // z
  next[2] = safety[2] / (abs(dir_local[2]) + kTiny);
  coord1 = pos_local[0] + next[2] * dir_local[0];
  coord2 = pos_local[1] + next[2] * dir_local[1];
  hit = safety[2] > 0 &&
        pos_local[2] * dir_local[2] < 0 &&
        (abs(coord1) <= dimensions[0] && abs(coord2) <= dimensions[1]);
  MaskedAssign(!done && hit, next[1], distance);

  return distance;
}

} // End namespace box 
} // End namespace kernel

#endif /* KERNEL_H */