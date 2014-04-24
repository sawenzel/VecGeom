#ifndef VECGEOM_PLACEDBOX_H_
#define VECGEOM_PLACEDBOX_H_

#include "PlacedVolume.h"
#include "UnplacedBox.h"
#include "Kernel.h"
#include "ShapeImplementationHelper.h"

class PlacedBox
    : public PlacedVolume,
      public ShapeImplementationHelper<PlacedBox, PlacedBox> {

private:

  typedef ShapeImplementationHelper<PlacedBox, PlacedBox> Implementation;

protected:

  UnplacedBox const *unplaced_;

public:

  PlacedBox(UnplacedBox const *const unplaced)
      : Implementation(this), unplaced_(unplaced) {}

  ~PlacedBox() {}

  VECGEOM_SHAPE_DISPATCH

  VECGEOM_SHAPE_IMPLEMENTATION

};

template <class Backend, class ShapeSpecialization>
void PlacedBox::InsideDispatch(typename Backend::double_v const point[3],
                               typename Backend::bool_v &output) const {
  BoxInside<Backend>(*unplaced_, point, output);
}

#endif // VECGEOM_PLACEDBOX_H_