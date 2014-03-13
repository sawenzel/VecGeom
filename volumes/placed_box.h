#ifndef VECGEOM_VOLUMES_PLACEDBOX_H_
#define VECGEOM_VOLUMES_PLACEDBOX_H_

#include "base/global.h"
#include "backend.h"
#include "volumes/placed_volume.h"
#include "volumes/unplaced_box.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

class PlacedBox : public VPlacedVolume {

public:

  VECGEOM_CUDA_HEADER_BOTH
  PlacedBox(LogicalVolume const *const logical_volume,
            TransformationMatrix const *const matrix)
      : VPlacedVolume(logical_volume, matrix, this) {}

  virtual ~PlacedBox() {}

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> const& dimensions() const {
    return AsUnplacedBox()->dimensions();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return AsUnplacedBox()->x(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return AsUnplacedBox()->y(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return AsUnplacedBox()->z(); }

protected:

  /**
   * Retrieves the unplaced volume pointer from the logical volume and casts it
   * to an unplaced box.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedBox const* AsUnplacedBox() const;
    virtual bool Inside(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
    virtual bool Inside(Vector3D<Precision> const &point, Vector3D<Precision> & localpoint) const;


public:

  // Navigation methods

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point) const;

  virtual void Inside(SOA3D<Precision> const &points,
                      bool *const output) const;

  virtual void Inside(AOS3D<Precision> const &points,
                      bool *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max) const;

  virtual void DistanceToIn(SOA3D<Precision> const &position,
                            SOA3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const;

  virtual void DistanceToIn(AOS3D<Precision> const &position,
                            AOS3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction) const;

  // CUDA specific

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   TransformationMatrix const *const matrix,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix) const;
  #endif

  // Comparison specific

  #ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
  virtual TGeoShape const* ConvertToRoot() const;
  virtual ::VUSolid const* ConvertToUSolids() const;
  #endif

  // Templates to interact with common kernel

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::bool_v InsideDispatch(
      Vector3D<typename Impl<it>::precision_v> const &point) const;

  template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Impl<it>::precision_v DistanceToInDispatch(
      Vector3D<typename Impl<it>::precision_v> const &position,
      Vector3D<typename Impl<it>::precision_v> const &direction,
      const typename Impl<it>::precision_v step_max) const;

};

template <TranslationCode trans_code, RotationCode rot_code, ImplType it>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Impl<it>::bool_v PlacedBox::InsideDispatch(
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
typename Impl<it>::precision_v PlacedBox::DistanceToInDispatch(
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

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PlacedBox::Inside(Vector3D<Precision> const &point) const {
  return PlacedBox::template InsideDispatch<1, 0, kScalar>(point);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::DistanceToIn(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction,
                                  const Precision step_max) const {
  return PlacedBox::template DistanceToInDispatch<1, 0, kScalar>(position,
                                                                 direction,
                                                                 step_max);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::DistanceToOut(Vector3D<Precision> const &position,
                                   Vector3D<Precision> const &direction) const {

  Vector3D<Precision> const &dim = AsUnplacedBox()->dimensions();

  const Vector3D<Precision> safety_plus  = dim + position;
  const Vector3D<Precision> safety_minus = dim - position;

  Vector3D<Precision> distance = safety_minus;
  const Vector3D<bool> direction_plus = direction < 0.0;
  distance.MaskedAssign(direction_plus, safety_plus);

  distance /= direction;

  const Precision min = distance.Min();
  return (min < 0) ? 0 : min;
}

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDBOX_H_
