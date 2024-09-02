#ifndef VECGEOM_SURFACE_DEVICESTORAGE_H_
#define VECGEOM_SURFACE_DEVICESTORAGE_H_

namespace vgbrep {

template <typename Real_t>
struct SurfData;
struct VolumeTree;

namespace globaldevicesurfdata {

/// @brief Storage for device pointer to surface data
/// @tparam Real_t Precision type
template <typename Real_t>
inline VECCORE_ATT_DEVICE SurfData<Real_t> *gSurfDataDevice = nullptr;

} // namespace globaldevicesurfdata
} // namespace vgbrep

#endif
