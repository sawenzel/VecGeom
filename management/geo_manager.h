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
template <typename Precision>
class GeoManager {

private:

  static int counter;
  static std::list<VPlacedVolume<Precision> const*> volumes;

public:

  static std::list<VPlacedVolume<Precision> const*> const& VolumeList() {
    return volumes;
  }

private:

  GeoManager() {
    counter = 0;
  }

  static int RegisterVolume(VPlacedVolume<Precision> const *const volume) {
    volumes.push_back(volume);
    return counter++;
  }

  /**
   * Deregistering will not change the counter, as gaps in the id don't have any
   * practical consequence.
   */
  static void DeregisterVolume(VPlacedVolume<Precision> const *const volume) {
    volumes.remove(volume);
  }

  friend VPlacedVolume<Precision>;

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_GEOMANAGER_BOXKERNEL_H_