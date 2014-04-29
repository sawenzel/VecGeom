#ifndef VECGEOM_PLACEDBOX_H_
#define VECGEOM_PLACEDBOX_H_

#include "PlacedVolume.h"
#include "UnplacedBox.h"

class PlacedBox : public PlacedVolume {

private:

  UnplacedBox const *fUnplacedBox;

public:

  typedef UnplacedBox UnplacedShape_t;

  PlacedBox(UnplacedBox const *const unplacedBox) : fUnplacedBox(unplacedBox) {}

  virtual ~PlacedBox() {}

  UnplacedBox const* GetUnplacedVolume() const { return fUnplacedBox; }

};

#endif // VECGEOM_PLACEDBOX_H_