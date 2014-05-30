/// \file RootGeoManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/Transformation3D.h"
#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedRootVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedRootVolume.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"

#include <cassert>

namespace vecgeom {

void RootGeoManager::LoadRootGeometry() {
  Clear();
  TGeoNode const *const world_root = ::gGeoManager->GetTopNode();
  // Convert() will recursively convert daughters
  world_ = Convert(world_root);
  GeoManager::Instance().set_world(world_);
}

VPlacedVolume* RootGeoManager::Convert(TGeoNode const *const node) {
  if (placed_volumes_.Contains(node)) return placed_volumes_[node];

  Transformation3D const *const transformation = Convert(node->GetMatrix());
  LogicalVolume *const logical_volume = Convert(node->GetVolume());
  VPlacedVolume *const placed_volume =
      logical_volume->Place(node->GetName(), transformation);

  int remaining_daughters = 0;
  {
    // All or no daughters should have been placed already
    remaining_daughters = node->GetNdaughters()
                          - logical_volume->daughters().size();
    assert(remaining_daughters == 0 ||
           remaining_daughters == node->GetNdaughters());
  }
  for (int i = 0; i < remaining_daughters; ++i) {
    logical_volume->PlaceDaughter(Convert(node->GetDaughter(i)));
  }

  placed_volumes_.Set(node, placed_volume);
  return placed_volume;
}

Transformation3D* RootGeoManager::Convert(TGeoMatrix const *const geomatrix) {
  if (transformations_.Contains(geomatrix)) return transformations_[geomatrix];

  Double_t const *const t = geomatrix->GetTranslation();
  Double_t const *const r = geomatrix->GetRotationMatrix();
  Transformation3D *const transformation =
      new Transformation3D(t[0], t[1], t[2], r[0], r[1], r[2],
                           r[3], r[4], r[5], r[6], r[7], r[8]);

  transformations_.Set(geomatrix, transformation);
  return transformation;
}

LogicalVolume* RootGeoManager::Convert(TGeoVolume const *const volume) {
  if (logical_volumes_.Contains(volume)) return logical_volumes_[volume];

  VUnplacedVolume const *const unplaced = Convert(volume->GetShape());
  LogicalVolume *const logical_volume =
      new LogicalVolume(volume->GetName(), unplaced);

  logical_volumes_.Set(volume, logical_volume);
  return logical_volume;
}

VUnplacedVolume* RootGeoManager::Convert(TGeoShape const *const shape) {
  if (unplaced_volumes_.Contains(shape)) return unplaced_volumes_[shape];

  VUnplacedVolume *unplaced_volume = NULL;
  if (shape->IsA() == TGeoBBox::Class()) {
    TGeoBBox const *const box = static_cast<TGeoBBox const*>(shape);
    unplaced_volume = new UnplacedBox(box->GetDX(), box->GetDY(), box->GetDZ());
  }
  // New volumes should be implemented here...
  if (!unplaced_volume) {
    if (verbose_) {
      printf("Unsupported shape for ROOT volume \"%s\". "
             "Using ROOT implementation.\n", shape->GetName());
    }
    unplaced_volume = new UnplacedRootVolume(shape);
  }
  
  unplaced_volumes_.Set(shape, unplaced_volume);
  return unplaced_volume;
}

void RootGeoManager::PrintNodeTable() const
{
   for(auto iter : placed_volumes_)
   {
      std::cerr << iter.first << " " << iter.second << "\n";
      TGeoNode const * n = iter.second;
      n->Print();
   }
}

void RootGeoManager::Clear() {
  for (auto i = placed_volumes_.begin(); i != placed_volumes_.end(); ++i) {
    delete i->first;
  }
  for (auto i = unplaced_volumes_.begin(); i != unplaced_volumes_.end(); ++i) {
    delete i->first;
  }
  for (auto i = logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {
    delete i->first;
  }
  for (auto i = transformations_.begin(); i != transformations_.end(); ++i) {
    delete i->first;
  }
  if (GeoManager::Instance().world() == world_) {
    GeoManager::Instance().set_world(nullptr);
  }
}

} // End global namespace
