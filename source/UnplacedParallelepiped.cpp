#include "volumes/UnplacedParallelepiped.h"

#include "management/volume_factory.h"
#include "volumes/SpecializedParallelepiped.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
UnplacedParallelepiped::UnplacedParallelepiped(
    Vector3D<Precision> const &dimensions,
    const Precision alpha,  const Precision theta, const Precision phi)
    : fDimensions(dimensions) {
  fTanAlpha = tan(alpha);
  fTanThetaCosPhi = tan(theta)*cos(phi);
  fTanThetaSinPhi = tan(theta)*sin(phi);
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedParallelepiped::UnplacedParallelepiped(
    const Precision x, const Precision y, const Precision z,
    const Precision alpha, const Precision theta, const Precision phi) {
  UnplacedParallelepiped(Vector3D<Precision>(x, y, z), alpha, theta, phi);
}

void UnplacedParallelepiped::Print() const {
  printf("UnplacedParallelepiped {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f}",
         GetX(), GetY(), GetZ(), GetTanAlpha(), GetTanThetaCosPhi(),
         GetTanThetaSinPhi());
}

void UnplacedParallelepiped::Print(std::ostream &os) const {
  os << "UnplacedParallelepiped {" << GetX() << ", " << GetY() << ", " << GetZ()
     << ", " << GetTanAlpha() << ", " << GetTanThetaCosPhi() << ", "
     << GetTanThetaSinPhi();
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParallelepiped::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  if (placement) {
    return new(placement) SpecializedParallelepiped<transCodeT, rotCodeT>(
        logical_volume, transformation
#ifdef VECGEOM_NVCC
        , const int id,
#endif
        );
  }
  return new SpecializedParallelepiped<transCodeT, rotCodeT>(
      logical_volume, transformation
#ifdef VECGEOM_NVCC
      , const int id
#endif
      );
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParallelepiped::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::Instance().CreateByTransformation<
      UnplacedParallelepiped>(volume, transformation, trans_code, rot_code,
                              placement);
}

} // End global namespace