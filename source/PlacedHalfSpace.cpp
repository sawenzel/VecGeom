#include "VecGeom/volumes/PlacedHalfSpace.h"

#include "VecGeom/volumes/SpecializedHalfSpace.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedHalfSpace::PrintType() const { printf("PlacedHalfSpace"); }

void PlacedHalfSpace::PrintType(std::ostream &os) const { os << "PlacedHalfSpace"; }

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedHalfSpace::ConvertToUnspecialized() const
{
  return new SimpleHalfSpace(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedHalfSpace)

#endif // VECCORE_CUDA

} // namespace vecgeom
