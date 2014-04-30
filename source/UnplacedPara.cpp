#include "volumes/UnplacedPara.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
UnplacedPara::UnplacedPara(const Precision x, const Precision y, 
                           const Precision z, const Precision alpha,
                           const Precision theta, const Precision phi)
    : fX(x), fY(y), fZ(z) {

  assert(x > 0 && y > 0 && z > 0 &&
         "UnplacedPara received invalid parameters.");

  SetAlpha(alpha);
  SetThetaAndPhi(theta, phi);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedPara::SetAlpha(const Precision alpha) {
  fTanAlpha = tan(alpha);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedPara::SetThetaAndPhi(const Precision theta, const Precision phi) {
  fTanThetaCosPhi = tan(theta)*cos(phi);
  fTanThetaSinPhi = tan(theta)*sin(phi);
}

#ifndef VECGEOM_NVCC

VPlacedVolume* SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement = NULL) const {
  return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                 placement);
}

#else

__device__
VPlacedVolume* SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement = NULL) const {
  return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                 id, placement);
}

#endif

} // End global namespace