#ifndef VECGEOM_PLACEDTUBE_H_
#define VECGEOM_PLACEDTUBE_H_

#include "PlacedVolume.h"
#include "UnplacedTube.h"

struct GeneralTube {
  static const bool isFancy = false;
};

struct FancyTube {
  static const bool isFancy = true;
};

class PlacedTube : public PlacedVolume {

protected:

  UnplacedTube const *fUnplacedTube;

public:

  typedef UnplacedTube UnplacedShape_t;

  PlacedTube(UnplacedTube const *const unplacedTube)
      : fUnplacedTube(unplacedTube) {}

  virtual ~PlacedTube() {}

  UnplacedTube const* GetUnplacedVolume() const { return fUnplacedTube; }

};

#endif // VECGEOM_PLACEDTUBE_H_