#ifndef VECGEOM_PLACEDVOLUME_H_
#define VECGEOM_PLACEDVOLUME_H_

#include "Global.h"

class PlacedVolume {

public:

  PlacedVolume() {}

  ~PlacedVolume() {}

  virtual bool Inside(const double point[3]) const =0;

  virtual void Inside(const double points[3][VcDouble::Size],
                      bool output[VcDouble::Size]) const =0;

};

#endif // VECGEOM_PLACEDVOLUME_H_