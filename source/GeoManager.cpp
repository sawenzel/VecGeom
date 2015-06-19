/// \file GeoManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/GeoManager.h"
#include "volumes/PlacedVolume.h"
#include <dlfcn.h>
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void GeoManager::RegisterLogicalVolume(LogicalVolume *const logical_volume) {
  fLogicalVolumesMap[logical_volume->id()] = logical_volume;
}

void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume) {
  fPlacedVolumesMap[placed_volume->id()] = placed_volume;
}

void GeoManager::DeregisterLogicalVolume(const int id) {
  fLogicalVolumesMap.erase(id);
}

void GeoManager::DeregisterPlacedVolume(const int id) {
  fPlacedVolumesMap.erase(id);
}

void GeoManager::CloseGeometry() {
    Assert( GetWorld() != NULL, "world volume not set" );
    // cache some important variables of this geometry
    GetMaxDepthVisitor depthvisitor;
    visitAllPlacedVolumes( GetWorld(), &depthvisitor, 1 );
    fMaxDepth = depthvisitor.getMaxDepth();

    GetTotalNodeCountVisitor totalcountvisitor;
    visitAllPlacedVolumes( GetWorld(), &totalcountvisitor, 1 );
    fTotalNodeCount = totalcountvisitor.GetTotalNodeCount();
}


void GeoManager::LoadGeometryFromSharedLib( std::string libname ){
    void *handle;
    handle = dlopen(libname.c_str(), RTLD_NOW);
    if (!handle){
        std::cerr << "Error loading geometry shared lib: " << dlerror() << "\n";
    }

    // the create detector "function type":
    typedef VPlacedVolume const * (*CreateFunc_t)();

    // find entry symbol to geometry creation
    // TODO: get rid of hard coded name
    CreateFunc_t create = (CreateFunc_t) dlsym(handle,"_Z16generateDetectorv");

    if (create != nullptr ){
      // call the create function and set the geometry world
      SetWorld( create() );

      // close the geometry
      // TODO: This step often necessitates extensive computation and could be done
      // as part of the shared lib load itself
      CloseGeometry();
    }
    else {
      std::cerr << "Loading geometry from shared lib failed\n";
    }
}


VPlacedVolume* GeoManager::FindPlacedVolume(const int id) {
  auto iterator = fPlacedVolumesMap.find(id);
  return (iterator != fPlacedVolumesMap.end()) ? iterator->second : NULL;
}

VPlacedVolume* GeoManager::FindPlacedVolume(char const *const label) {
  VPlacedVolume *output = NULL;
  bool multiple = false;
  for (auto v = fPlacedVolumesMap.begin(), v_end = fPlacedVolumesMap.end();
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
  auto iterator = fLogicalVolumesMap.find(id);
  return (iterator != fLogicalVolumesMap.end()) ? iterator->second : NULL;
}

LogicalVolume* GeoManager::FindLogicalVolume(char const *const label) {
  LogicalVolume *output = NULL;
  bool multiple = false;
  for (auto v = fLogicalVolumesMap.begin(), v_end = fLogicalVolumesMap.end();
       v != v_end; ++v) {

    const std::string& fullname = v->second->GetLabel();
    if (fullname.compare(label)==0) {
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

void GeoManager::Clear()
{
    fLogicalVolumesMap.clear();
    fPlacedVolumesMap.clear();
    fVolumeCount=0; fWorld=NULL;
    fMaxDepth=-1;
}

} } // End global namespace
