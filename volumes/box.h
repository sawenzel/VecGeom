#ifndef VECGEOM_VOLUMES_BOX_H_
#define VECGEOM_VOLUMES_BOX_H_

#include "volumes/unplaced_volume.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

template <typename Precision>
class UnplacedBox : VUnplacedVolume<Precision> {};

template <typename Precision>
class PlacedBox : VPlacedVolume<Precision> {};

template <typename Precision>
class SpecializedBox : PlacedBox<Precision> {};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_BOX_H_