#ifndef VECGEOM_SPECIALIZEDTUBE_H_
#define VECGEOM_SPECIALIZEDTUBE_H_

#include "PlacedTube.h"

template <class TubeSpecialization>
class SpecializedTube
    : public PlacedTube,
      public ShapeImplementation<PlacedTube, TubeSpecialization> {
  SpecializedTube(UnplacedTube const *const unplaced) : PlacedTube(unplaced) {}
};

#endif // VECGEOM_SPECIALIZEDTUBE_H_