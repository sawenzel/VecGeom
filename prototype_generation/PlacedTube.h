#ifndef VECGEOM_PLACEDTUBE_H_
#define VECGEOM_PLACEDTUBE_H_

#include "PlacedVolume.h"
#include "UnplacedTube.h"
#include "Kernel.h"
#include "ShapeImplementationHelper.h"

struct GeneralTube {
  static const bool is_fancy = false;
};

struct FancyTube {
  static const bool is_fancy = true;
};

class PlacedTube : public PlacedVolume,
                   public ShapeImplementationHelper<PlacedTube, PlacedTube> {

private:

  typedef ShapeImplementationHelper<PlacedTube, PlacedTube> Implementation;

protected:

  UnplacedTube const *unplaced_;

public:

  // This is a problem... so far necessary to avoid ambiguity
  static const bool is_fancy = false;

  PlacedTube(UnplacedTube const *const unplaced)
      : Implementation(this), unplaced_(unplaced) {}

  ~PlacedTube() {}

  VECGEOM_SHAPE_DISPATCH

  VECGEOM_SHAPE_IMPLEMENTATION

};

template <class Backend, class ShapeSpecialization>
void PlacedTube::InsideDispatch(typename Backend::double_v const point[3],
                                typename Backend::bool_v &output) const {
  TubeInside<Backend, ShapeSpecialization>(*unplaced_, point, output);
}

#endif // VECGEOM_PLACEDTUBE_H_