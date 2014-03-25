/**
 * @file placed_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <stdio.h>

#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::PrintContent(const int depth) const {
  for (int i = 0; i < depth; ++i) printf("  ");
  printf("%i: ", id_);
  unplaced_volume()->Print();
  printf("\n");
  for (Iterator<VPlacedVolume const*> vol = daughters().begin();
       vol != daughters().end(); ++vol) {
    (*vol)->PrintContent(depth + 1);
  }
}

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol) {
  os << "(" << (*vol.unplaced_volume()) << ", " << (*vol.matrix()) << ")";
  return os;
}

} // End global namespace