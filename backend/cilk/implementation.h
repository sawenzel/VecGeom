/**
 * @file cilk/implementation.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_CILK_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_CILK_IMPLEMENTATION_H_

#include "base/global.h"

#include "backend/cilk/backend.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#include "volumes/placed_box.h"
#include "volumes/kernel/box_kernel.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode trans_code, RotationCode rot_code,
          typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::Inside_Looper(VolumeType const &volume,
                                  ContainerType const &points,
                                  bool *const output) {
  for (int i = 0; i < points.fillsize(); i += kVectorSize) {
    const CilkBool result =
        volume.template InsideDispatch<trans_code, rot_code, kCilk>(
          Vector3D<CilkPrecision>(CilkPrecision(&points.x(i)),
                                  CilkPrecision(&points.y(i)),
                                  CilkPrecision(&points.z(i)))
        );
    result.store(&output[i]);
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
    const CilkPrecision result =
        volume.template DistanceToInDispatch<trans_code, rot_code,
                                                          kCilk>(
          Vector3D<CilkPrecision>(CilkPrecision(&positions.x(i)),
                                  CilkPrecision(&positions.y(i)),
                                  CilkPrecision(&positions.z(i))),
          Vector3D<CilkPrecision>(CilkPrecision(&directions.x(i)),
                                  CilkPrecision(&directions.y(i)),
                                  CilkPrecision(&directions.z(i))),
          CilkPrecision(&step_max[i])
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
    const CilkPrecision result =
        volume.template DistanceToOutDispatch<kCilk>(
          Vector3D<CilkPrecision>(CilkPrecision(&positions.x(i)),
                                  CilkPrecision(&positions.y(i)),
                                  CilkPrecision(&positions.z(i))),
          Vector3D<CilkPrecision>(CilkPrecision(&directions.x(i)),
                                  CilkPrecision(&directions.y(i)),
                                  CilkPrecision(&directions.z(i))),
          CilkPrecision(&step_max[i])
        );
    result.store(&output[i]);
  }
}

template <TranslationCode trans_code, RotationCode rot_code,
          typename VolumeType, typename ContainerType>
VECGEOM_INLINE
void VPlacedVolume::SafetyToIn_Looper(VolumeType const &volume,
                                      ContainerType const &positions,
                                      Precision *const output) {
  for (int i = 0; i < positions.fillsize(); i += kVectorSize) {
    const CilkPrecision result =
        volume.template SafetyToInDispatch<trans_code,rot_code, kCilk>(
          Vector3D<CilkPrecision>(CilkPrecision(&positions.x(i)),
                                  CilkPrecision(&positions.y(i)),
                                  CilkPrecision(&positions.z(i)))
        );
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
    const CilkPrecision result =
        volume.template SafetyToOutDispatch<kCilk>(
          Vector3D<CilkPrecision>(CilkPrecision(&positions.x(i)),
                                  CilkPrecision(&positions.y(i)),
                                  CilkPrecision(&positions.z(i)))
        );
    result.store(&output[i]);
  }
}


} // End global namespace

#endif // VECGEOM_BACKEND_CILK_IMPLEMENTATION_H_
