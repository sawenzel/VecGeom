#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/types.h"
#include "base/transformation_matrix.h"
#include "management/geo_manager.h"

namespace vecgeom {

template <typename Precision>
class VPlacedVolume {

private:

  int id;

protected:

  VUnplacedVolume<Precision> const &unplaced_volume_;
  TransformationMatrix<Precision> const &matrix_;

public:

  VPlacedVolume(VUnplacedVolume<Precision> const &unplaced_volume__,
                TransformationMatrix<Precision> const &matrix__)
      : unplaced_volume_(unplaced_volume__), matrix_(matrix__) {
    id = GeoManager<Precision>::Instance().RegisterVolume(this);
  }

  ~VPlacedVolume() {
    GeoManager<Precision>::Instance().DeregisterVolume(this);
  }

  VLogicalVolume<Precision> const& unplaced_volume() const {
    return unplaced_volume_;
  }

  TransformationMatrix<Precision> const& matrix() const {
    return matrix_;
  }

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_]