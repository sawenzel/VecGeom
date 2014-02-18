#ifndef VECGEOM_MANAGEMENT_GEOMANAGER_BOXKERNEL_H_
#define VECGEOM_MANAGEMENT_GEOMANAGER_BOXKERNEL_H_

#include <iostream>
#include <list>
#include "base/types.h"

namespace vecgeom {

/**
 * Singleton class that maintains a list of all instatiated placed volumes.
 * Will assign each placed volume a unique id that identifies them globally.
 */
class GeoManager {

private:

  int counter;
  std::list<VPlacedVolume const*> volumes_;

public:

  static GeoManager& Instance() {
    static GeoManager instance;
    return instance;
  }

  std::list<VPlacedVolume const*> const& volumes() {
    return volumes_;
  }

private:

  GeoManager() {
    counter = 0;
  }

  GeoManager(GeoManager const&);
  GeoManager& operator=(GeoManager const&);

  int RegisterVolume(VPlacedVolume const *const volume) {
    volumes_.push_back(volume);
    return counter++;
  }

  /**
   * Deregistering will not change the counter, as gaps in the id don't have any
   * practical consequence.
   */
  void DeregisterVolume(VPlacedVolume const *const volume) {
    volumes_.remove(volume);
  }

  friend VPlacedVolume;

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_GEOMANAGER_BOXKERNEL_H_