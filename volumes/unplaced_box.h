/**
 * @file unplaced_box.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDBOX_H_
#define VECGEOM_VOLUMES_UNPLACEDBOX_H_

#include <iostream>
#include "base/global.h"
#include "base/vector3d.h"
#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedBox : public VUnplacedVolume {

private:

  Vector3D<Precision> dimensions_;

public:

  UnplacedBox(Vector3D<Precision> const &dim) {
    dimensions_ = dim;
  }

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(const Precision dx, const Precision dy, const Precision dz)
      : dimensions_(dx, dy, dz) {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(UnplacedBox const &other) : dimensions_(other.dimensions_) {}

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
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

  template <TranslationCode trans_code, RotationCode rot_code>
  VECGEOM_CUDA_HEADER_BOTH
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               TransformationMatrix const *const matrix,
                               VPlacedVolume *const placement = NULL);

  VECGEOM_CUDA_HEADER_BOTH
  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      TransformationMatrix const *const matrix,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL);
  
private:

  VECGEOM_CUDA_HEADER_BOTH
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      TransformationMatrix const *const matrix,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, matrix, trans_code, rot_code,
                                   placement);
  }

  virtual void Print(std::ostream &os) const {
    os << "Box {" << dimensions_ << "}";
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDBOX_H_