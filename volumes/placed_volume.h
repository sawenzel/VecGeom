#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/types.h"
#include "management/geo_manager.h"
#include "volumes/logical_volume.h"
#include "base/transformation_matrix.h"

namespace vecgeom {

template <typename Precision>
class VPlacedVolume {

protected:

  const int id;
  VLogicalVolume<Precision> const &volume;
  TransformationMatrix<Precision> const &matrix;

public:

  VPlacedVolume(VLogicalVolume<Precision> const &volume_,
                TransformationMatrix<Precision> const &matrix_) {
    id = GeoManager<Precision>::template RegisterVolume(this);
    volume = volume_;
    matrix = matrix_;
  }

  ~VPlacedVolume() {
    GeoManager<Precision>::template DeregisterVolume(this);
  }

  VLogicalVolume<Precision> const& Volume() const { return volume; }

  TransformationMatrix<Precision> const& Matrix() const { return matrix; }

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_]