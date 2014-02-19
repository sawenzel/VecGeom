#include "volumes/unplaced_volume.h"

namespace vecgeom {

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol) {
  vol.print(os);
  return os;
}

} // End namespace vecgeom