#ifndef VECGEOM_MANAGEMENT_ROOTMANAGER_H_
#define VECGEOM_MANAGEMENT_ROOTMANAGER_H_

#include "base/global.h"

#include "base/type_map.h"

namespace vecgeom {

class RootManager {

private:

  TypeMap<VPlacedVolume const*, TGeoNode const*> placed_volumes_;
  TypeMap<VUnplacedVolume const*, TGeoShape const*> unplaced_volumes_;
  TypeMap<LogicalVolume const*, TGeoVolume const*> logical_volumes_;
  TypeMap<TransformationMatrix const*, TGeoMatrix const*> matrices_;

public:

  static RootManager& Instance() {
    static RootManager instance;
    return instance;
  }

  VPlacedVolume* Convert(TGeoNode const *const node);

  VUnplacedVolume* Convert(TGeoShape const *const shape);

  LogicalVolume* Convert(TGeoVolume const *const volume);

  UnplacedBox* Convert(TGeoBBox const *const box);

  TransformationMatrix* Convert(TGeoMatrix const *const matrix);

private:

  RootManager() {}
  RootManager(RootManager const&);
  RootManager& operator=(RootManager const&);

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_ROOTMANAGER_H_