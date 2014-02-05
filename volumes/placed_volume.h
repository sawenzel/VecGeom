#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "volumes/logical_volume.h"
#include "base/trans_matrix.h"

namespace vecgeom {

template <typename Precision>
class VPlacedVolume {

protected:

  VLogicalVolume<Precision> const &volume;
  TransMatrix<Precision> const &matrix;

public:

  VPlacedVolume(VLogicalVolume<Precision> const &volume_,
                TransMatrix<Precision> const &matrix_) {
    volume = volume_;
    matrix = matrix_;
  }

  VLogicalVolume<Precision> const& Volume() const { return volume; }

  TransMatrix<Precision> const& Matrix() const { return matrix; }

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_]