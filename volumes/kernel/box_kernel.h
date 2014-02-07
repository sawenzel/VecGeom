#ifndef VECGEOM_VOLUMES_KERNEL_BOXKERNEL_H_
#define VECGEOM_VOLUMES_KERNEL_BOXKERNEL_H_

#include "base/types.h"
#include "base/vector3d.h"
#include "base/transformation_matrix.h"
#include "base/utilities.h"

namespace vecgeom {

template <ImplType it, TranslationCode trans_code, RotationCode rot_code>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxInside(Vector3D<typename Impl<it>::precision> const &dimensions,
               TransformationMatrix<typename Impl<it>::precision> const &matrix,
               Vector3D<typename Impl<it>::double_v> const &point,
               typename Impl<it>::bool_v *const inside) {

  const Vector3D<typename Impl<it>::double_v> local =
      matrix.template Transform<trans_code, rot_code>(point);

  Vector3D<typename Impl<it>::bool_v> inside_dim(Impl<it>::kFalse);
  for (int i = 0; i < 3; ++i) {
    inside_dim[i] = Abs<it>(local[i]) < dimensions[i];
    if (Impl<it>::early_returns) {
      if (!inside_dim[i]) {
        *inside = Impl<it>::kFalse;
        return;
      }
    }
  }

  if (Impl<it>::early_returns) {
    *inside = Impl<it>::kTrue;
  } else {
    *inside = inside_dim[0] && inside_dim[1] && inside_dim[2];
  }
}

template <ImplType it, TranslationCode trans_code, RotationCode rot_code>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxDistanceToIn(
    Vector3D<typename Impl<it>::precision> const &dimensions,
    TransformationMatrix<typename Impl<it>::precision> const &matrix,
    Vector3D<typename Impl<it>::double_v> const &pos,
    Vector3D<typename Impl<it>::double_v> const &dir,
    typename Impl<it>::double_v const &step_max,
    typename Impl<it>::double_v *const distance) {

  typedef typename Impl<it>::double_v Float;
  typedef typename Impl<it>::bool_v Bool;

  Vector3D<Float> safety;
  Vector3D<Float> pos_local;
  Vector3D<Float> dir_local;
  Bool hit(false);
  Bool done(false);
  *distance = kInfinity;

  matrix.template Transform<trans_code, rot_code>(pos, &pos_local);
  matrix.template TransformRotation<rot_code>(dir, &dir_local);

  safety[0] = Abs<it>(pos_local[0]) - dimensions[0];
  safety[1] = Abs<it>(pos_local[1]) - dimensions[1];
  safety[2] = Abs<it>(pos_local[2]) - dimensions[2];

  done |= (safety[0] >= step_max ||
           safety[1] >= step_max ||
           safety[2] >= step_max);
  if (done == true) return;

  Float next, coord1, coord2;

  // x
  next = safety[0] / Abs<it>(dir_local[0] + kTiny);
  coord1 = pos_local[1] + next * dir_local[1];
  coord2 = pos_local[2] + next * dir_local[2];
  hit = safety[0] > 0 &&
        pos_local[0] * dir_local[0] < 0 &&
        Abs<it>(coord1) <= dimensions[1] &&
        Abs<it>(coord2) <= dimensions[2];
  MaskedAssign<it>(!done && hit, next, distance);
  done |= hit;
  if (done == true) return;

  // y
  next = safety[1] / Abs<it>(dir_local[1] + kTiny);
  coord1 = pos_local[0] + next * dir_local[0];
  coord2 = pos_local[2] + next * dir_local[2];
  hit = safety[1] > 0 &&
        pos_local[1] * dir_local[1] < 0 &&
        Abs<it>(coord1) <= dimensions[0] &&
        Abs<it>(coord2) <= dimensions[2];
  MaskedAssign<it>(!done && hit, next, distance);
  done |= hit;
  if (done == true) return;

  // z
  next = safety[2] / Abs<it>(dir_local[2] + kTiny);
  coord1 = pos_local[0] + next * dir_local[0];
  coord2 = pos_local[1] + next * dir_local[1];
  hit = safety[2] > 0 &&
        pos_local[2] * dir_local[2] < 0 &&
        Abs<it>(coord1) <= dimensions[0] &&
        Abs<it>(coord2) <= dimensions[1];
  MaskedAssign<it>(!done && hit, next, distance);

}

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_BOXKERNEL_H_