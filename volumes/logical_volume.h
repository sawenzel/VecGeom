#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include "base/types.h"
#include "base/vector.h"
#include "volumes/unplaced_volume.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

template <typename Precision>
class VLogicalVolume {

private:

  VUnplacedVolume<Precision> const &unplaced_volume_;
  Container<VPlacedVolume<Precision> const*> *daughters;

  friend VPlacedVolume<Precision>;

public:

  VECGEOM_CUDA_HEADER_HOST
  VLogicalVolume(VUnplacedVolume<Precision> const &unplaced_volume__)
      : unplaced_volume_(unplaced_volume__) {
    daughters = new Vector<VPlacedVolume<Precision> const*>();
  }

  VECGEOM_CUDA_HEADER_HOST
  ~VLogicalVolume() {
    Vector<VPlacedVolume<Precision> const*> *vector =
        static_cast<Vector<VPlacedVolume<Precision> const*> *>(daughters);
    for (int i = 0; i < vector->size(); ++i) {
      delete (*vector)[i];
    }
    delete vector;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume<Precision> const &unplaced_volume() const {
    return unplaced_volume_;
  }

  VECGEOM_CUDA_HEADER_HOST
  void PlaceDaughter(VUnplacedVolume<Precision> const &volume,
                     TransformationMatrix<Precision> const &matrix) {
    VPlacedVolume<Precision> *placed =
        new VPlacedVolume<Precision>(volume, matrix);
    static_cast<Vector<VPlacedVolume<Precision> const*> *>(
      daughters
    )->push_back(placed);
  }

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_