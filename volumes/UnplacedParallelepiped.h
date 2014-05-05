#ifndef VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_

#include "base/global.h"

#include "base/AlignedBase.h"
#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedParallelepiped : public VUnplacedVolume, AlignedBase {

private:

  Vector3D<Precision> fDimensions;
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
  Precision GetTanAlpha() const { return fTanAlpha; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const { return fTanThetaSinPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const { return fTanThetaCosPhi; }

  virtual int memory_size() const { return sizeof(*this); }

  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

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

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_