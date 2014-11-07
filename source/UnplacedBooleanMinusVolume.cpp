/*
 * UnplacedBooleanMinusVolume.cpp
 *
 *  Created on: 07.11.2014
 *      Author: swenzel
 */

#include "base/Global.h"
#include "volumes/UnplacedBooleanMinusVolume.h"
#include "volumes/SpecializedBooleanMinusVolume.h"
#include "management/VolumeFactory.h"
#include "volumes/utilities/GenerationUtilities.h"

namespace VECGEOM_NAMESPACE
{


template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedBooleanMinusVolume::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  return CreateSpecializedWithPlacement<SpecializedBooleanMinusVolume<transCodeT, rotCodeT> >(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedBooleanMinusVolume::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<UnplacedBooleanMinusVolume>(
    volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
    id,
#endif
    placement);
}


}
