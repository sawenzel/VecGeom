#ifndef VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedParallelepiped; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedParallelepiped );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedParallelepiped : public VUnplacedVolume, public AlignedBase {

private:

  Vector3D<Precision> fDimensions;
  Precision fAlpha, fTheta, fPhi;

  // Precomputed values computed from parameters
  Precision fTanAlpha, fTanThetaSinPhi, fTanThetaCosPhi;

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped(Vector3D<Precision> const &dimensions,
                         const Precision alpha, const Precision theta,
                         const Precision phi);

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped(const Precision x, const Precision y,
                         const Precision z, const Precision alpha,
                         const Precision theta, const Precision phi);

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const& GetDimensions() const { return fDimensions; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetX() const { return fDimensions[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetY() const { return fDimensions[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZ() const { return fDimensions[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha() const { return fAlpha; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return fTheta; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return fPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha() const { return fTanAlpha; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const { return fTanThetaSinPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const { return fTanThetaCosPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  void SetDimensions(Vector3D<Precision> const &dimensions) {
    fDimensions = dimensions;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetDimensions(const Precision x, const Precision y, const Precision z) {
    fDimensions.Set(x, y, z);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetAlpha(const Precision alpha);

  VECGEOM_CUDA_HEADER_BOTH
  void SetTheta(const Precision theta);

  VECGEOM_CUDA_HEADER_BOTH
  void SetPhi(const Precision phi);

  VECGEOM_CUDA_HEADER_BOTH
  void SetThetaAndPhi(const Precision theta, const Precision phi);

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedParallelepiped>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
