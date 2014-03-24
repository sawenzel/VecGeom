/**
 * @file vc/implementation.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_VC_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_VC_IMPLEMENTATION_H_

#include "base/global.h"

#include "backend/vc/backend.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode trans_code, RotationCode rot_code,
          typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::Inside_Looper(VolumeType const &volume,
                                  ContainerType const &points,
                                  bool *const output) {
  for (int i = 0; i < points.fillsize(); i += kVectorSize) {
    const VcBool result =
        volume.template InsideDispatch<trans_code, rot_code, kVc>(
          Vector3D<VcPrecision>(VcPrecision(&points.ContainerType::x(i)),
                                VcPrecision(&points.ContainerType::y(i)),
                                VcPrecision(&points.ContainerType::z(i)))
        );
    for (int j = 0; j < kVectorSize; ++j) {
      output[j] = result[j];
    }
  }
}

template <TranslationCode trans_code, RotationCode rot_code,
          typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::DistanceToIn_Looper(VolumeType const &volume,
                                        ContainerType const &positions,
                                        ContainerType const &directions,
                                        Precision const *const step_max,
                                        Precision *const output) {
  for (int i = 0; i < positions.fillsize(); i += kVectorSize) {
    const VcPrecision result =
        volume.template DistanceToInDispatch<trans_code, rot_code, kVc>(
          Vector3D<VcPrecision>(VcPrecision(&positions.ContainerType::x(i)),
                                VcPrecision(&positions.ContainerType::y(i)),
                                VcPrecision(&positions.ContainerType::z(i))),
          Vector3D<VcPrecision>(VcPrecision(&directions.ContainerType::x(i)),
                                VcPrecision(&directions.ContainerType::y(i)),
                                VcPrecision(&directions.ContainerType::z(i))),
          VcPrecision(&step_max[i])
        );
    result.store(&output[i]);
  }
}

template <typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::DistanceToOut_Looper(VolumeType const &volume,
                                        ContainerType const &positions,
                                        ContainerType const &directions,
                                        Precision const *const step_max,
                                        Precision *const output) {
  for (int i = 0; i < positions.fillsize(); i += kVectorSize) {
    const VcPrecision result =
        volume.template DistanceToOutDispatch<kVc>(
          Vector3D<VcPrecision>(VcPrecision(&positions.ContainerType::x(i)),
                                VcPrecision(&positions.ContainerType::y(i)),
                                VcPrecision(&positions.ContainerType::z(i))),
          Vector3D<VcPrecision>(VcPrecision(&directions.ContainerType::x(i)),
                                VcPrecision(&directions.ContainerType::y(i)),
                                VcPrecision(&directions.ContainerType::z(i))),
          VcPrecision(&step_max[i])
        );
    result.store(&output[i]);
  }
}

template <TranslationCode trans_code, RotationCode rot_code,
			typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::SafetyToIn_Looper(VolumeType const &volume,
                                      ContainerType const &positions,
                                      Precision *const output)
{
  for (int i = 0; i < positions.fillsize(); i += kVectorSize) {
    const VcPrecision result =
        volume.template SafetyToInDispatch<trans_code,rot_code, kVc>(
          Vector3D<VcPrecision>(VcPrecision(&positions.ContainerType::x(i)),
                                VcPrecision(&positions.ContainerType::y(i)),
                                VcPrecision(&positions.ContainerType::z(i))));
    	result.store(&output[i]);
  }
}

template <typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::SafetyToOut_Looper(VolumeType const &volume,
                                      ContainerType const &positions,
                                      Precision *const output)
{
  for (int i = 0; i < positions.fillsize(); i += kVectorSize) {
    const VcPrecision result =
        volume.template SafetyToOutDispatch<kVc>(
          Vector3D<VcPrecision>(VcPrecision(&positions.ContainerType::x(i)),
                                VcPrecision(&positions.ContainerType::y(i)),
                                VcPrecision(&positions.ContainerType::z(i))));
    	result.store(&output[i]);
  }
}


} // End global namespace

#endif // VECGEOM_BACKEND_VC_IMPLEMENTATION_H_
