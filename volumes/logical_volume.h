#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include <vector>
#include "base/types.h"

namespace vecgeom {

template <typename Precision>
class VLogicalVolume {

private:

  VUnplacedVolume<Precision> const &volume;
  // Will potentially be a custom container class
  std::vector<VPlacedVolume<Precision> const*> daughters;

public:

  VLogicalVolume(VUnplacedVolume<Precision> const &volume_) {
    volume = volume_;
  }

  void PlaceDaughter(VPlacedVolume<Precision> const *const daughter) {
    daughters.push_back(daughter);
  }

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_