#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include <vector>
#include "base/types.h"
#include "base/list.h"

namespace vecgeom {

template <typename Precision>
class VLogicalVolume {

private:

  VUnplacedVolume<Precision> const &volume;
  Container<VPlacedVolume<Precision> const*> *daughters;

public:

  VECGEOM_CUDA_HEADER_HOST
  VLogicalVolume(VUnplacedVolume<Precision> const &volume_) {
    volume = volume_;
    daughters = new List<VPlacedVolume<Precision> const*>();
  }

  VECGEOM_CUDA_HEADER_HOST
  ~VLogicalVolume() {
    delete daughters;
  }

  VECGEOM_CUDA_HEADER_HOST
  void PlaceDaughter(VPlacedVolume<Precision> const *const daughter) {
    dynamic_cast<List<VPlacedVolume<Precision> const*> >(daughters)->push_back(
      daughter
    );
  }

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_