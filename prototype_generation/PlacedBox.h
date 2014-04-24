#ifndef VECGEOM_PLACEDBOX_H_
#define VECGEOM_PLACEDBOX_H_

#include "PlacedVolume.h"
#include "UnplacedBox.h"
#include "Kernel.h"
#include "ShapeImplementation.h"

class PlacedBox
    : public PlacedVolume,
      public ShapeImplementation<PlacedBox, PlacedBox> {

private:

  typedef ShapeImplementation<PlacedBox, PlacedBox> Implementation;

  UnplacedBox const *unplaced_;

public:

  PlacedBox(UnplacedBox const *const unplaced) : unplaced_(unplaced) {}

  ~PlacedBox() {}

  UnplacedBox const* unplaced() const { return unplaced_; }

  template <class Backend, class BoxSpecialization>
  void InsideDispatch(typename Backend::double_v const point[3],
                      typename Backend::bool_v &output) const {
    BoxInside<Backend>(*unplaced_, point, output);
  }

  VECGEOM_INSIDE_IMPLEMENTATION

};

#endif // VECGEOM_PLACEDBOX_H_