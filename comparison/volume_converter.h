#ifndef VECGEOM_COMPARISON_VOLUMECONVERTER_H_
#define VECGEOM_COMPARISON_VOLUMECONVERTER_H_

#include "base/global.h"

class TGeoShape;
class VUSolid;

namespace vecgeom {

class VolumeConverter {

private:

  VPlacedVolume const *vecgeom_;
  TGeoShape const *root_;
  ::VUSolid const *usolids_;

public:

  VolumeConverter(VPlacedVolume const *const volume);
  ~VolumeConverter();

  VPlacedVolume const* vecgeom() const { return vecgeom_; }

  TGeoShape const* root() const { return root_; }

  ::VUSolid const* usolids() const { return usolids_; }

};

} // End namespace vecgeom

#endif // VECGEOM_COMPARISON_VOLUMECONVERTER_H_