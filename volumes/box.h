#ifndef VECGEOM_VOLUMES_BOX_H_
#define VECGEOM_VOLUMES_BOX_H_

#include "base/transformation_matrix.h"
#include "volumes/unplaced_volume.h"
#include "volumes/placed_volume.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

template <typename Precision>
class UnplacedBox : public VUnplacedVolume<Precision> {

private:

  Vector3D<Precision> dimensions_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(Vector3D<Precision> const &dim) {
    dimensions_ = dim;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> const& dimensions() const { return dimensions_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return dimensions_[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return dimensions_[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return dimensions_[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision volume() const {
    return 4.0*dimensions_[0]*dimensions_[1]*dimensions_[2];
  }

};

template <typename Precision>
class PlacedBox : public VPlacedVolume<Precision> {

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::bool_v Inside(Vector3D<Precision> const &point) const;

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::float_v DistanceToIn(Vector3D<Precision> const &position,
                                          Vector3D<Precision> const &direction,
                                          const Precision step_max) const;

};

template <TranslationCode trans_code, RotationCode rot_code, typename Precision>
class SpecializedBox : public PlacedBox<Precision> {

  template <ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::bool_v Inside(Vector3D<Precision> const &point) const {
    PlacedBox<Precision>::template Inside<trans_code, rot_code, it>(point);
  }

  template <ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::float_v DistanceToIn(Vector3D<Precision> const &position,
                                          Vector3D<Precision> const &direction,
                                          const Precision step_max) const {
    PlacedBox<Precision>::template DistanceToIn<trans_code, rot_code, it>(
      position, direction, step_max
    );
  }

};

template <typename Precision>
template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
typename Impl<it>::bool_v PlacedBox<Precision>::Inside(
    Vector3D<Precision> const &point) const {

  typename Impl<it>::bool_v output;

  BoxInside<trans_code, rot_code, it>(
    static_cast<UnplacedBox<Precision> >(this->unplaced_volume_)->dimensions(),
    this->matrix,
    point,
    &output
  );

  return output;
}

template <typename Precision>
template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
typename Impl<it>::float_v PlacedBox<Precision>::DistanceToIn(
    Vector3D<Precision> const &position,
    Vector3D<Precision> const &direction,
    const Precision step_max) const {

  typename Impl<it>::float_v output;

  BoxDistanceToIn<trans_code, rot_code, it>(
    static_cast<UnplacedBox<Precision> >(this->unplaced_volume_)->dimensions(),
    this->matrix,
    position,
    direction,
    step_max,
    &output
  );

  return output;
}

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_BOX_H_