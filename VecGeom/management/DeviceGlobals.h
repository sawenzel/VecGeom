/// \file DeviceGlobals.h
/// \author Andrei Gheata 23/08/2024

#ifndef VECGEOM_MANAGEMENT_DEVICEGLOBALS_H_
#define VECGEOM_MANAGEMENT_DEVICEGLOBALS_H_

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct VolumeTree;);

struct VolumeTree;

// we put some global data into a separate namespace
// this is done since CUDA does not support static const members in class definitions
namespace globaldevicegeomdata {
inline VECCORE_ATT_DEVICE VolumeTree *gVolumeTree = nullptr;
} // namespace globaldevicegeomdata
} // namespace vecgeom
#endif
