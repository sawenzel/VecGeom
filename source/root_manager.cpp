/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "base/transformation_matrix.h"
#include "management/geo_manager.h"
#include "management/root_manager.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"
#include "volumes/unplaced_box.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"

namespace vecgeom {

void RootManager::LoadRootGeometry() {
  Clear();
  TGeoNode const *const world_root = ::gGeoManager->GetTopNode();
  world_ = Convert(world_root);
  GeoManager::Instance().set_world(world_);
}

VPlacedVolume* RootManager::Convert(TGeoNode const *const node) {
  if (placed_volumes_.Contains(node)) return placed_volumes_[node];

  TransformationMatrix const *const matrix = Convert(node->GetMatrix());
  LogicalVolume *const logical_volume = Convert(node->GetVolume());
  VPlacedVolume *const placed_volume = logical_volume->Place(matrix);
  for (int i = 0; i < node->GetNdaughters(); ++i) {
    logical_volume->PlaceDaughter(Convert(node->GetDaughter(i)));
  }

  placed_volumes_.Set(node, placed_volume);
  return placed_volume;
}

TransformationMatrix* RootManager::Convert(TGeoMatrix const *const geomatrix) {
  if (matrices_.Contains(geomatrix)) return matrices_[geomatrix];

  Double_t const *const t = geomatrix->GetTranslation();
  Double_t const *const r = geomatrix->GetRotationMatrix();
  TransformationMatrix *const matrix =
      new TransformationMatrix(t[0], t[1], t[2], r[0], r[1], r[2],
                               r[3], r[4], r[5], r[6], r[7], r[8]);

  matrices_.Set(geomatrix, matrix);
  return matrix;
}

LogicalVolume* RootManager::Convert(TGeoVolume const *const volume) {
  if (logical_volumes_.Contains(volume)) return logical_volumes_[volume];

  VUnplacedVolume const *const unplaced = Convert(volume->GetShape());
  LogicalVolume *const logical_volume = new LogicalVolume(unplaced);

  logical_volumes_.Set(volume, logical_volume);
  return logical_volume;
}

VUnplacedVolume* RootManager::Convert(TGeoShape const *const shape) {
  if (unplaced_volumes_.Contains(shape)) return unplaced_volumes_[shape];

  // Dynamic casts are necessary to avoid tight coupling
  VUnplacedVolume *unplaced_volume = NULL;
  if (TGeoBBox const *box = dynamic_cast<TGeoBBox const*>(shape)) {
    unplaced_volume = new UnplacedBox(box->GetDX(), box->GetDY(), box->GetDZ());
  }
  if (!unplaced_volume) {
    std::cerr << "Attempted to convert unsupported shape.\n";
    assert(unplaced_volume);
  }
  
  unplaced_volumes_.Set(shape, unplaced_volume);
  return unplaced_volume;
}

void RootManager::Clear() {
  for (auto i = placed_volumes_.begin(); i != placed_volumes_.end(); ++i) {
    delete i->first;
  }
  for (auto i = unplaced_volumes_.begin(); i != unplaced_volumes_.end(); ++i) {
    delete i->first;
  }
  for (auto i = logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {
    delete i->first;
  }
  for (auto i = matrices_.begin(); i != matrices_.end(); ++i) {
    delete i->first;
  }
  if (GeoManager::Instance().world() == world_) {
    GeoManager::Instance().set_world(nullptr);
  }
}

} // End namespace vecgeom