#include "volumes/placed_volume.h"

namespace vecgeom {

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol) {
  os << "(" << vol.unplaced_volume() << ", " << vol.matrix() << ")";
  return os;
}

} // End namespace vecgeom