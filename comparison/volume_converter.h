#ifndef VECGEOM_COMPARISON_VOLUMECONVERTER_H_
#define VECGEOM_COMPARISON_VOLUMECONVERTER_H_

#include "base/global.h"

class TGeoShape;
class VUSolid;

namespace vecgeom {

class VolumeConverter {

private:

  VPlacedVolume const *specialized_;
  VPlacedVolume const *unspecialized_;
  TGeoShape const *root_;
  ::VUSolid const *usolids_;

public:

  VolumeConverter(VPlacedVolume const *const volume);

  VolumeConverter(VolumeConverter const &other);

  ~VolumeConverter();

  VolumeConverter& operator=(VolumeConverter const &other);

  VPlacedVolume const* specialized() const { return specialized_; }

  VPlacedVolume const* unspecialized() const { return unspecialized_; }

  TGeoShape const* root() const { return root_; }

  ::VUSolid const* usolids() const { return usolids_; }

private:

  void ConvertVolume();
  
  void Deallocate();

};

} // End namespace vecgeom

#endif // VECGEOM_COMPARISON_VOLUMECONVERTER_H_