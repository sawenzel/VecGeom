#ifndef VECGEOM_MANAGEMENT_GEOMANAGER_H_
#define VECGEOM_MANAGEMENT_GEOMANAGER_H_

#include "base/global.h"

namespace vecgeom {

/**
 * Singleton class that maintains a list of all instatiated placed volumes.
 * Will assign each placed volume a unique id that identifies them globally.
 */
class GeoManager {

public:

  static GeoManager& Instance() {
    static GeoManager instance;
    return instance;
  }

private:

  GeoManager() {}

  GeoManager(GeoManager const&);
  GeoManager& operator=(GeoManager const&);

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_GEOMANAGER_H_