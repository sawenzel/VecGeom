#ifndef VECGEOM_SPECIALIZEDTUBE_H_
#define VECGEOM_SPECIALIZEDTUBE_H_

#include "Kernel.h"
#include "PlacedTube.h"
#include "ShapeImplementationHelper.h"

template <class TubeSpecialization>
class SpecializedTube
    : public ShapeImplementationHelper<
                 PlacedTube,
                 TubeImplementation<TubeSpecialization> > {

public:

  SpecializedTube(UnplacedTube const *const unplacedTube)
      : ShapeImplementationHelper<
                 PlacedTube,
                 TubeImplementation<TubeSpecialization> >(unplacedTube) {}

};

#endif // VECGEOM_SPECIALIZEDTUBE_H_