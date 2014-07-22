/// \file GeoManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/GeoManager.h"

#include "volumes/PlacedVolume.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

void GeoManager::RegisterLogicalVolume(LogicalVolume *const logical_volume) {
  logical_volumes_[logical_volume->id()] = logical_volume;
}

void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume) {
  placed_volumes_[placed_volume->id()] = placed_volume;
}

void GeoManager::DeregisterLogicalVolume(const int id) {
  logical_volumes_.erase(id);
}

void GeoManager::DeregisterPlacedVolume(const int id) {
  placed_volumes_.erase(id);
}

void GeoManager::CloseGeometry() {
   // cache some important variables of this geometry
    GetMaxDepthVisitor depthvisitor;
    visitAllPlacedVolumes( world(), &depthvisitor, 1 );
    fMaxDepth = depthvisitor.getMaxDepth();
}



VPlacedVolume* GeoManager::FindPlacedVolume(const int id) {
  auto iterator = placed_volumes_.find(id);
  return (iterator != placed_volumes_.end()) ? iterator->second : NULL;
}

VPlacedVolume* GeoManager::FindPlacedVolume(char const *const label) {
  VPlacedVolume *output = NULL;
  bool multiple = false;
  for (auto v = placed_volumes_.begin(), v_end = placed_volumes_.end();
       v != v_end; ++v) {
    if (v->second->GetLabel() == label) {
      if (!output) {
        output = v->second;
      } else {
        if (!multiple) {
          multiple = true;
          printf("GeoManager::FindPlacedVolume: Multiple placed volumes with "
                 "identifier \"%s\" found: [%i], ", label, output->id());
        } else {
          printf(", ");
        }
        printf("[%i]", v->second->id());
      }
    }
  }
  if (multiple) printf(". Returning first occurence.\n");
  return output;
}

LogicalVolume* GeoManager::FindLogicalVolume(const int id) {
  auto iterator = logical_volumes_.find(id);
  return (iterator != logical_volumes_.end()) ? iterator->second : NULL;
}

LogicalVolume* GeoManager::FindLogicalVolume(char const *const label) {
  LogicalVolume *output = NULL;
  bool multiple = false;
  for (auto v = logical_volumes_.begin(), v_end = logical_volumes_.end();
       v != v_end; ++v) {
    if (v->second->GetLabel() == label) {
      if (!output) {
        output = v->second;
      } else {
        if (!multiple) {
          multiple = true;
          printf("GeoManager::FindLogicalVolume: Multiple logical volumes with "
                 "identifier \"%s\" found: [%i], ", label, output->id());
        } else {
          printf(", ");
        }
        printf("[%i]", v->second->id());
      }
    }
  }
  if (multiple) printf(". Returning first occurence.\n");
  return output;
}

} // End global namespace
