#ifndef VECGEOM_VOLUMES_BOX_H_
#define VECGEOM_VOLUMES_BOX_H_

#include "base/transformation_matrix.h"
#include "volumes/unplaced_volume.h"
#include "volumes/placed_volume.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

class UnplacedBox : public VUnplacedVolume {

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

private:

  virtual void print(std::ostream &os) const {
    os << "Box {" << dimensions_ << "}";
  }

};

class PlacedBox : public VPlacedVolume {

public:

  #if (!defined(VECGEOM_INTEL) && defined(VECGEOM_STD_CXX11))
  using VPlacedVolume::VPlacedVolume;
  #else
  PlacedBox(UnplacedBox const &unplaced_volume__,
            TransformationMatrix const &matrix__)
      : VPlacedVolume(unplaced_volume__, matrix__) {}
  #endif

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::bool_v Inside(
      Vector3D<typename Impl<it>::precision_v> const &point) const;

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::precision_v DistanceToIn(
      Vector3D<typename Impl<it>::precision_v> const &position,
      Vector3D<typename Impl<it>::precision_v> const &direction,
      const typename Impl<it>::precision_v step_max) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision DistanceToOut(Vector3D<Precision> const &position,
                          Vector3D<Precision> const &direction) const;

private:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedBox const& AsUnplacedBox() const;

};

template <TranslationCode trans_code, RotationCode rot_code>
class SpecializedBox : public PlacedBox {

  template <ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::bool_v Inside(
      Vector3D<typename Impl<it>::precision_v> const &point) const {
    PlacedBox::template Inside<trans_code, rot_code, it>(point);
  }

  template <ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::precision_v DistanceToIn(
      Vector3D<typename Impl<it>::precision_v> const &position,
      Vector3D<typename Impl<it>::precision_v> const &direction,
      const typename Impl<it>::precision_v step_max) const {
    PlacedBox::template DistanceToIn<trans_code, rot_code, it>(
      position, direction, step_max
    );
  }

};

template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Impl<it>::bool_v PlacedBox::Inside(
    Vector3D<typename Impl<it>::precision_v> const &point) const {

  typename Impl<it>::bool_v output;

  BoxInside<trans_code, rot_code, it>(
    AsUnplacedBox().dimensions(),
    this->matrix(),
    point,
    &output
  );

  return output;
}

template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Impl<it>::precision_v PlacedBox::DistanceToIn(
    Vector3D<typename Impl<it>::precision_v> const &position,
    Vector3D<typename Impl<it>::precision_v> const &direction,
    const typename Impl<it>::precision_v step_max) const {

  typename Impl<it>::precision_v output;

  BoxDistanceToIn<trans_code, rot_code, it>(
    AsUnplacedBox().dimensions(),
    this->matrix(),
    position,
    direction,
    step_max,
    &output
  );

  return output;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::DistanceToOut(
    Vector3D<Precision> const &position,
    Vector3D<Precision> const &direction) const {

  Vector3D<Precision> const &dim =
      AsUnplacedBox().dimensions();

  Vector3D<Precision> const safety_plus  = dim + position;
  Vector3D<Precision> const safety_minus = dim - position;

  Vector3D<Precision> distance = safety_minus;
  distance.MaskedAssign(direction < 0.0, safety_plus);

  distance /= direction;

  const Precision min = distance.Min();
  return (min < 0) ? 0 : min;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
UnplacedBox const& PlacedBox::AsUnplacedBox() const {
  return static_cast<UnplacedBox const&>(this->unplaced_volume());
}

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_BOX_H_