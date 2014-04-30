#ifndef VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_

#include "base/global.h"

#include "base/AlignedBase.h"
#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

template <bool HasTanAlpha, bool HasTanThetaSinPhi, bool HasTanThetaCosPhi,
          TranslationCode tTransCode, RotationCode tRotCode>
struct ParallelepipedSpecialization {};

class UnplacedParallelepiped : public AlignedBase {

private:

  Vector3D<Precision> fDimensions;
  Precision fTanAlpha, fTanThetaSinPhi, fTanThetaCosPhi;

public:

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

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_