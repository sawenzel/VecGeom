#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

VECGEOM_CUDA_HEADER_HOST
VLogicalVolume::~VLogicalVolume() {
  Vector<VPlacedVolume const*> *vector =
      static_cast<Vector<VPlacedVolume const*> *>(daughters_);
  for (int i = 0; i < vector->size(); ++i) {
    delete (*vector)[i];
  }
  delete vector;
}

VECGEOM_CUDA_HEADER_HOST
void VLogicalVolume::PlaceDaughter(VLogicalVolume const &volume,
                                   TransformationMatrix const &matrix) {
  VPlacedVolume *placed = new VPlacedVolume(volume, matrix);
  static_cast<Vector<VPlacedVolume const*> *>(
    daughters_
  )->push_back(placed);
}

} // End namespace vecgeom