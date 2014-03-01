#ifndef VECGEOM_VOLUMES_BOX_H_
#define VECGEOM_VOLUMES_BOX_H_

#include <stdio.h>
#include "base/transformation_matrix.h"
#include "backend/scalar_backend.h"
#include "volumes/unplaced_volume.h"
#include "volumes/placed_volume.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

class UnplacedBox : public VUnplacedVolume {

private:

  Vector3D<Precision> dimensions_;

public:

  UnplacedBox(Vector3D<Precision> const &dim) {
    dimensions_ = dim;
  }

  UnplacedBox(const Precision dx, const Precision dy, const Precision dz)
      : dimensions_(dx, dy, dz) {}

  virtual int byte_size() const { return sizeof(*this); }

  #ifdef VECGEOM_NVCC
  virtual void CopyToGpu(VUnplacedVolume *const target) const;
  #endif

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

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;
  
private:

  virtual void Print(std::ostream &os) const {
    os << "Box {" << dimensions_ << "}";
  }

};

class PlacedBox : public VPlacedVolume {

public:

  #if (!defined(VECGEOM_INTEL) && defined(VECGEOM_STD_CXX11))
  using VPlacedVolume::VPlacedVolume;
  #else
  PlacedBox(LogicalVolume const *const logical_volume__,
            TransformationMatrix const *const matrix__)
      : VPlacedVolume(logical_volume__, matrix__) {}
  #endif

  virtual ~PlacedBox() {}

  virtual int byte_size() const { return sizeof(*this); }

  // Virtual volume methods

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction) const;

protected:

  // Templates to interact with common kernel

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::bool_v InsideTemplate(
      Vector3D<typename Impl<it>::precision_v> const &point) const;

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::precision_v DistanceToInTemplate(
      Vector3D<typename Impl<it>::precision_v> const &position,
      Vector3D<typename Impl<it>::precision_v> const &direction,
      const typename Impl<it>::precision_v step_max) const;

  /**
   * Retrieves the unplaced volume pointer from the logical volume and casts it
   * to an unplaced box.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedBox const* AsUnplacedBox() const;

};

template <TranslationCode trans_code, RotationCode rot_code>
class SpecializedBox : public PlacedBox {

public:

  #if (!defined(VECGEOM_INTEL) && defined(VECGEOM_STD_CXX11))
  using PlacedBox::PlacedBox;
  #else
  SpecializedBox(LogicalVolume const *const logical_volume__,
                 TransformationMatrix const *const matrix__)
      : PlacedBox(logical_volume__, matrix__) {}
  #endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max) const;

  virtual int byte_size() const { return sizeof(*this); }

};

template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Impl<it>::bool_v PlacedBox::InsideTemplate(
    Vector3D<typename Impl<it>::precision_v> const &point) const {

  typename Impl<it>::bool_v output;

  BoxInside<trans_code, rot_code, it>(
    AsUnplacedBox()->dimensions(),
    *this->matrix(),
    point,
    &output
  );

  return output;
}

template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Impl<it>::precision_v PlacedBox::DistanceToInTemplate(
    Vector3D<typename Impl<it>::precision_v> const &position,
    Vector3D<typename Impl<it>::precision_v> const &direction,
    const typename Impl<it>::precision_v step_max) const {

  typename Impl<it>::precision_v output;

  BoxDistanceToIn<trans_code, rot_code, it>(
    AsUnplacedBox()->dimensions(),
    *this->matrix(),
    position,
    direction,
    step_max,
    &output
  );

  return output;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
UnplacedBox const* PlacedBox::AsUnplacedBox() const {
  return static_cast<UnplacedBox const*>(this->unplaced_volume());
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedBox<trans_code, rot_code>::Inside(
    Vector3D<Precision> const &point) const {
  return PlacedBox::template InsideTemplate<trans_code, rot_code, kScalar>(
           point
         );
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedBox<trans_code, rot_code>::DistanceToIn(
    Vector3D<Precision> const &position,
    Vector3D<Precision> const &direction,
    const Precision step_max) const {

  return PlacedBox::template DistanceToInTemplate<trans_code, rot_code,
                                                  kScalar>(position, direction,
                                                           step_max);
                                                  
}

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_BOX_H_