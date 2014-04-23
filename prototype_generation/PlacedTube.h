#ifndef VECGEOM_PLACEDTUBE_H_
#define VECGEOM_PLACEDTUBE_H_

#include "PlacedVolume.h"
#include "UnplacedTube.h"
#include "Kernel.h"
#include "ShapeImplementation.h"

struct GeneralTube {
  static const bool is_fancy = false;
};

struct FancyTube {
  static const bool is_fancy = true;
};

class PlacedTube : public ShapeImplementation<PlacedTube, GeneralTube> {

private:

  UnplacedTube const *unplaced_;

public:

  PlacedTube(UnplacedTube const *const unplaced) : unplaced_(unplaced) {}

  ~PlacedTube() {}

  UnplacedTube const* unplaced() const { return unplaced_; }

  template <class Backend, class TubeSpecialization>
  void InsideDispatch(typename Backend::double_v const point[3],
                      typename Backend::bool_v &output) const {
    TubeInside<Backend, TubeSpecialization>(*unplaced_, point, output);
  }

};

#endif // VECGEOM_PLACEDTUBE_H_