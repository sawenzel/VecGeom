/// @file UnplacedBox.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDBOX_H_
#define VECGEOM_VOLUMES_UNPLACEDBOX_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedBox; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedBox );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedBox : public VUnplacedVolume, public AlignedBase {

private:

  Vector3D<Precision> dimensions_;

public:

  UnplacedBox(Vector3D<Precision> const &dim) : dimensions_(dim) { }

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(const Precision dx, const Precision dy, const Precision dz)
      : dimensions_(dx, dy, dz) {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(UnplacedBox const &other) : dimensions_(other.dimensions_) {}

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedBox>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
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
    return 8.0*dimensions_[0]*dimensions_[1]*dimensions_[2];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const {
     // factor 8 because dimensions_ are half-lengths
     return 8.0*(dimensions_[0]*dimensions_[1] +
             dimensions_[1]*dimensions_[2] + dimensions_[2]*dimensions_[0]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &, Vector3D<Precision> &) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetPointOnSurface() const;


  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

#ifndef VECGEOM_NVCC

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL);

#else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               const int id,
                               VPlacedVolume *const placement = NULL);

  __device__
  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL);

#endif
  
private:

#ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   placement);
  }

#else

  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   id, placement);
  }

#endif
  
};

} } // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDBOX_H_
