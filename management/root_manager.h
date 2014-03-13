/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_MANAGEMENT_ROOTMANAGER_H_
#define VECGEOM_MANAGEMENT_ROOTMANAGER_H_

#include "base/global.h"

#include "base/type_map.h"

namespace vecgeom {

class RootManager {

private:

  VPlacedVolume const* world_ = nullptr;

  TypeMap<VPlacedVolume*, TGeoNode const*> placed_volumes_;
  TypeMap<VUnplacedVolume*, TGeoShape const*> unplaced_volumes_;
  TypeMap<LogicalVolume*, TGeoVolume const*> logical_volumes_;
  TypeMap<TransformationMatrix*, TGeoMatrix const*> matrices_;

public:

  static RootManager& Instance() {
    static RootManager instance;
    return instance;
  }

  VPlacedVolume const* world() const { return world_; }

  /**
   * Will register the imported ROOT geometry as the new world of the VecGeom
   * GeoManager singleton.
   */
  void LoadRootGeometry();

  void Clear();

  VPlacedVolume* Convert(TGeoNode const *const node);

  VUnplacedVolume* Convert(TGeoShape const *const shape);

  LogicalVolume* Convert(TGeoVolume const *const volume);

  UnplacedBox* Convert(TGeoBBox const *const box);

  TransformationMatrix* Convert(TGeoMatrix const *const matrix);

private:

  RootManager() {}
  RootManager(RootManager const&);
  RootManager& operator=(RootManager const&);

  VPlacedVolume* TraverseVolume(TGeoNode const *const node);

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_ROOTMANAGER_H_