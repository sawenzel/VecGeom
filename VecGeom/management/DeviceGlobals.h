/// \file DeviceGlobals.h
/// \author Andrei Gheata 23/08/2024

#ifndef VECGEOM_MANAGEMENT_DEVICEGLOBALS_H_
#define VECGEOM_MANAGEMENT_DEVICEGLOBALS_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#pragma once
namespace vecgeom {

namespace cuda {
class VPlacedVolume;
class LogicalVolume;
} // namespace cuda

VECGEOM_DEVICE_FORWARD_DECLARE(struct VolumeTree;);
VECGEOM_DEVICE_FORWARD_DECLARE(class LogicalVolume;);
VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);

struct VolumeTree;

// we put some global data into a separate namespace
// this is done since CUDA does not support static const members in class definitions
namespace globaldevicegeomdata {
inline VECCORE_ATT_DEVICE VolumeTree *gVolumeTree                      = nullptr;
inline VECCORE_ATT_DEVICE cuda::VPlacedVolume *gCompactPlacedVolBuffer = nullptr;
inline VECCORE_ATT_DEVICE cuda::LogicalVolume *gDeviceLogicalVolumes   = nullptr;
inline VECCORE_ATT_DEVICE NavIndex_t *gNavIndex                        = nullptr; // address of navigation index table
inline VECCORE_ATT_DEVICE int gMaxDepth                                = 0;
} // namespace globaldevicegeomdata
} // namespace vecgeom
#endif
