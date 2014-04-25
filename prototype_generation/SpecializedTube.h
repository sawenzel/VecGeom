#ifndef VECGEOM_SPECIALIZEDTUBE_H_
#define VECGEOM_SPECIALIZEDTUBE_H_

#include "PlacedTube.h"

template <class TubeSpecialization>
class SpecializedTube
    : public PlacedTube,
      private ShapeImplementationHelper<PlacedTube, TubeSpecialization> {

private:

  typedef ShapeImplementationHelper<PlacedTube, TubeSpecialization>
      Implementation;

public:

  SpecializedTube(UnplacedTube const *const unplaced)
      : Implementation(this), PlacedTube(unplaced) {}

  ~SpecializedTube() {}

  VECGEOM_SHAPE_IMPLEMENTATION

};

#endif // VECGEOM_SPECIALIZEDTUBE_H_