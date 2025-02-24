#ifndef VECGEOM_SURFACE_SCALEDCONVERTER_H_
#define VECGEOM_SURFACE_SCALEDCONVERTER_H_

#include <VecGeom/volumes/ScaledShape.h>

// Contains only forward declarations, the implementation needs to call recursively CreateSolidSurfaces
// Implementation sits in SolidConverter.h

namespace vgbrep {
namespace conv {

template <typename Real_t>
bool CreateScaledSurfaces(vecgeom::UnplacedScaledShape const &scaled, int logical_id, bool intersection = false);

} // namespace conv
} // namespace vgbrep
#endif
