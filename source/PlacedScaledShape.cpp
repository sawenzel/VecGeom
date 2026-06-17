/// \file PlacedScaledShape.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/PlacedScaledShape.h"
#include "VecGeom/volumes/SpecializedScaledShape.h"

#include <stdio.h>

#ifdef VECGEOM_ROOT
#include "TGeoScaledShape.h"
#include "TGeoMatrix.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedScaledShape::PrintType() const { printf("PlacedScaledShape"); }

void PlacedScaledShape::PrintType(std::ostream &os) const { os << "PlacedScaledShape"; }

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedScaledShape::ConvertToUnspecialized() const
{
  return new SimpleScaledShape(GetLabel().c_str(), logical_volume_, GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedScaledShape::ConvertToRoot() const
{
  UnplacedScaledShape const *unplaced = const_cast<UnplacedScaledShape *>(GetUnplacedVolume());
  auto *rootShape                     = (TGeoShape *)unplaced->fScaled.fPlaced->ConvertToRoot();
  if (!rootShape) return nullptr;
  return new TGeoScaledShape(GetLabel().c_str(), rootShape,
                             new TGeoScale(unplaced->fScaled.fScale.Scale()[0], unplaced->fScaled.fScale.Scale()[1],
                                           unplaced->fScaled.fScale.Scale()[2]));
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedScaledShape)

#endif // VECCORE_CUDA

} // End namespace vecgeom
