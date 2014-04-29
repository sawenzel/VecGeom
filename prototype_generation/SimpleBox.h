#ifndef VECGEOM_SIMPLEBOX_H_
#define VECGEOM_SIMPLEBOX_H_

#include "Kernel.h"
#include "PlacedBox.h"
#include "ShapeImplementationHelper.h"

class SimpleBox
    : public ShapeImplementationHelper<PlacedBox, BoxImplementation> {

public:

  SimpleBox(UnplacedBox const *const unplacedBox)
      : ShapeImplementationHelper<PlacedBox,
                                  BoxImplementation>(unplacedBox) {}

};

#endif // VECGEOM_SIMPLEBOX_H_