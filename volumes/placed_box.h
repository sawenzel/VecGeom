#ifndef VECGEOM_VOLUMES_PLACEDBOX_H_
#define VECGEOM_VOLUMES_PLACEDBOX_H_

#include "base/global.h"
#include "backend/scalar_backend.h"
#include "base/transformation_matrix.h"
#include "volumes/placed_volume.h"
#include "volumes/unplaced_box.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

class PlacedBox : public VPlacedVolume {

public:

  #if (!defined(VECGEOM_INTEL) && defined(VECGEOM_STD_CXX11))
  using VPlacedVolume::VPlacedVolume;
  #else
  VECGEOM_CUDA_HEADER_BOTH
  PlacedBox(LogicalVolume const *const logical_volume__,
            TransformationMatrix const *const matrix__)
      : VPlacedVolume(logical_volume__, matrix__) {}
  #endif

  virtual ~PlacedBox() {}

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   TransformationMatrix const *const matrix,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix) const;
  #endif

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

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDBOX_H_