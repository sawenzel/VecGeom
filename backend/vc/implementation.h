/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_VC_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_VC_IMPLEMENTATION_H_

#include "base/global.h"

#include "backend/vc/backend.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

template <TranslationCode trans_code, RotationCode rot_code,
          typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::InsideBackend(VolumeType const &volume,
                                  ContainerType const &points,
                                  bool *const output) {
  for (int i = 0; i < points.size(); i += kVectorSize) {
    const VcBool result =
        volume.template InsideDispatch<trans_code, rot_code, kVc>(
          Vector3D<VcPrecision>(VcPrecision(&points.x(i)),
                                VcPrecision(&points.y(i)),
                                VcPrecision(&points.z(i)))
        );
    for (int j = 0; j < kVectorSize; ++j) {
      output[j] = result[j];
    }
  }
}

template <TranslationCode trans_code, RotationCode rot_code,
          typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::DistanceToInBackend(VolumeType const &volume,
                                        ContainerType const &positions,
                                        ContainerType const &directions,
                                        Precision const *const step_max,
                                        Precision *const output) {
  for (int i = 0; i < positions.size(); i += kVectorSize) {
    const VcPrecision result =
        volume.template DistanceToInDispatch<trans_code, rot_code, kVc>(
          Vector3D<VcPrecision>(VcPrecision(&positions.x(i)),
                                VcPrecision(&positions.y(i)),
                                VcPrecision(&positions.z(i))),
          Vector3D<VcPrecision>(VcPrecision(&directions.x(i)),
                                VcPrecision(&directions.y(i)),
                                VcPrecision(&directions.z(i))),
          VcPrecision(&step_max[i])
        );
    result.store(&output[i]);
  }
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_VC_IMPLEMENTATION_H_