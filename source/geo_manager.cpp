#include "management/geo_manager.h"
#include "volumes/placed_volume.h"

#include <stdio.h>

namespace vecgeom {

int GeoManager::getMaxDepth( ) const
{
   // walk all the volume hierarchy and insert
   // placed volumes if not already in the container
   GetMaxDepthVisitor depthvisitor;
   visitAllPlacedVolumes( world(), &depthvisitor, 1 );
   return depthvisitor.getMaxDepth();
}

VPlacedVolume* GeoManager::FindPlacedVolume(const int id) {
  for (auto v = VPlacedVolume::volume_list().begin(),
       v_end = VPlacedVolume::volume_list().end(); v != v_end; ++v) {
    if ((*v)->id() == id) return *v;
  }
  return NULL;
}

VPlacedVolume* GeoManager::FindPlacedVolume(char const *const label) {
  VPlacedVolume *output = NULL;
  bool multiple = false;
  for (auto v = VPlacedVolume::volume_list().begin(),
       v_end = VPlacedVolume::volume_list().end(); v != v_end; ++v) {
    if ((*v)->label() == label) {
      if (!output) {
        output = *v;
      } else {
        if (!multiple) {
          multiple = true;
          printf("GeoManager::FindPlacedVolume: Multiple placed volumes with "
                 "identifier \"%s\" found: [%i], ", label, output->id());
        } else {
          printf(", ");
        }
        printf("[%i]", (*v)->id());
      }
    }
  }
  if (multiple) printf(". Returning first occurence.\n");
  return output;
}

LogicalVolume* GeoManager::FindLogicalVolume(const int id) {
  for (auto v = LogicalVolume::volume_list().begin(),
       v_end = LogicalVolume::volume_list().end(); v != v_end; ++v) {
    if ((*v)->id() == id) return *v;
  }
  return NULL;
}

LogicalVolume* GeoManager::FindLogicalVolume(char const *const label) {
  LogicalVolume *output = NULL;
  bool multiple = false;
  for (auto v = LogicalVolume::volume_list().begin(),
       v_end = LogicalVolume::volume_list().end(); v != v_end; ++v) {
    if ((*v)->label() == label) {
      if (!output) {
        output = *v;
      } else {
        if (!multiple) {
          multiple = true;
          printf("GeoManager::FindLogicalVolume: Multiple logical volumes with "
                 "identifier \"%s\" found: [%i], ", label, output->id());
        } else {
          printf(", ");
        }
        printf("[%i]", (*v)->id());
      }
    }
  }
  if (multiple) printf(". Returning first occurence.\n");
  return output;
}

} // End global namespace
