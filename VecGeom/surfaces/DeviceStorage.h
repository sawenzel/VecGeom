#ifndef VECGEOM_SURFACE_DEVICESTORAGE_H_
#define VECGEOM_SURFACE_DEVICESTORAGE_H_

namespace vgbrep {

template <typename Real_t>
struct SurfData;

namespace globaldevicesurfdata {

/// @brief Storage for device pointer tu surface data
/// @tparam Real_t Precision type
template <typename Real_t>
inline VECCORE_ATT_DEVICE SurfData<Real_t> *gSurfDataDevice = nullptr;

} // namespace globaldevicesurfdata
} // namespace vgbrep

#endif
