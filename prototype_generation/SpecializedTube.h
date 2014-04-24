#ifndef VECGEOM_SPECIALIZEDTUBE_H_
#define VECGEOM_SPECIALIZEDTUBE_H_

#include "PlacedTube.h"

template <class TubeSpecialization>
class SpecializedTube
    : public PlacedTube,
      public ShapeImplementation<PlacedTube, TubeSpecialization> {

private:

  typedef ShapeImplementation<PlacedTube, TubeSpecialization> Implementation;

public:

  SpecializedTube(UnplacedTube const *const unplaced)
      : Implementation(this), PlacedTube(unplaced) {}

  VECGEOM_INSIDE_IMPLEMENTATION

};

#endif // VECGEOM_SPECIALIZEDTUBE_H_