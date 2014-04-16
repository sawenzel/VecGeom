#include "management/geo_manager.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

int GeoManager::getMaxDepth( ) const
{
   // walk all the volume hierarchy and insert
   // placed volumes if not already in the container
   GetMaxDepthVisitor depthvisitor;
   visitAllPlacedVolumes( world(), &depthvisitor, 1 );
   return depthvisitor.getMaxDepth();
}

VPlacedVolume* GeoManager::FindVolume(const int id) {
  for (auto v = VPlacedVolume::volume_list().begin(),
       v_end = VPlacedVolume::volume_list().end(); v != v_end; ++v) {
    if ((*v)->id() == id) return *v;
  }
  return NULL;
}

VPlacedVolume* GeoManager::FindVolume(char const *const label) {
  for (auto v = VPlacedVolume::volume_list().begin(),
       v_end = VPlacedVolume::volume_list().end(); v != v_end; ++v) {
    if ((*v)->label() == label) return *v;
  }
  return NULL;
}

} // End global namespace
