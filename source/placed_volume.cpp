/**
 * @file placed_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol) {
  os << "(" << (*vol.unplaced_volume()) << ", " << (*vol.matrix()) << ")";
  return os;
}

} // End global namespace