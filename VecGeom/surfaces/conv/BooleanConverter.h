#ifndef VECGEOM_SURFACE_BOOLEANCONVERTER_H_
#define VECGEOM_SURFACE_BOOLEANCONVERTER_H_

#include <VecGeom/volumes/BooleanVolume.h>
#include <VecGeom/surfaces/conv/LogicHelper.h>

// Contains only forward declarations, the implementation needs to call recursively CreateSolidSurfaces
// Implementation sits in SolidConverter.h

namespace vgbrep {
namespace conv {

template <typename Real_t>
bool CreateBooleanSurfaces(vecgeom::BooleanStruct const &bstruct, int logical_id, bool intersection = false);

template <typename Real_t>
bool AppendLogicTo(vecgeom::BooleanStruct const &bstruct, TransformationMP<Real_t> const &trans, int logical_id,
                   bool intersection = false);

} // namespace conv
} // namespace vgbrep
#endif
