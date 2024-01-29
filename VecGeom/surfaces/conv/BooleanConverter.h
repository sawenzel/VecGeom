#ifndef VECGEOM_SURFACE_BOOLEANCONVERTER_H_
#define VECGEOM_SURFACE_BOOLEANCONVERTER_H_

#include <VecGeom/volumes/BooleanVolume.h>
#include <VecGeom/surfaces/conv/LogicHelper.h>

// Contains only forward declarations, the implementation needs to call recursively CreateSolidSurfaces
// Implementation sits in SolidConverter.h

namespace vgbrep {
namespace conv {

template <typename Real_t>
bool CreateBooleanSurfaces(vecgeom::BooleanStruct const &bstruct, int logical_id);

template <typename Real_t>
void AppendLogicTo(vecgeom::BooleanStruct const &bstruct, Transformation const &trans, int logical_id);

} // namespace conv
} // namespace vgbrep
#endif
