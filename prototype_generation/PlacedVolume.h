#ifndef VECGEOM_PLACEDVOLUME_H_
#define VECGEOM_PLACEDVOLUME_H_

#include "Global.h"

class PlacedVolume {

public:

  PlacedVolume() {}

  ~PlacedVolume() {}

  virtual bool Inside(Vector3D<double> const &point) const =0;

  virtual void Inside(double const *const *const points, const int n,
                      bool *const output) const =0;

};

#endif // VECGEOM_PLACEDVOLUME_H_