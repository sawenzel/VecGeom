/**
 * @file geo_manager.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_MANAGEMENT_GEOMANAGER_H_
#define VECGEOM_MANAGEMENT_GEOMANAGER_H_

#include "base/global.h"

namespace VECGEOM_NAMESPACE {

/**
 * @brief Knows about the current world volume.
 */
class GeoManager {

private:

  VPlacedVolume const *world_;

public:

  static GeoManager& Instance() {
    static GeoManager instance;
    return instance;
  }

  void set_world(VPlacedVolume const *const world) { world_ = world; }

  VPlacedVolume const* world() const { return world_; }

private:

  GeoManager() {}

  GeoManager(GeoManager const&);
  GeoManager& operator=(GeoManager const&);

};

} // End global namespace

#endif // VECGEOM_MANAGEMENT_GEOMANAGER_H_