#ifndef VECGEOM_SURFACE_HELPER_H
#define VECGEOM_SURFACE_HELPER_H

#include <VecGeom/surfaces/CommonTypes.h>
#include <VecGeom/base/Vector3D.h>

namespace vgbrep {

/// @brief Default type for surface helpers
/// @tparam Real_t Precision type
/// @tparam Stype Surface type
template <SurfaceType Stype, typename Real_t>
struct SurfaceHelper {
};

} // namespace vgbrep

#endif