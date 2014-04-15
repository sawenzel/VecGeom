/**
 * @file placed_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <stdio.h>

#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

int VPlacedVolume::g_id_count = 0;
std::list<VPlacedVolume *> VPlacedVolume::g_volume_list =
    std::list<VPlacedVolume *>();

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::PrintContent(const int depth) const {
  for (int i = 0; i < depth; ++i) printf("  ");
  printf("[%i]", id_);
#ifndef VECGEOM_NVCC
  if (label_->size()) {
    printf(" \"%s\"", label_->c_str());
  }
#endif
  printf(": ");
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

VPlacedVolume* VPlacedVolume::FindVolume(const int id) {
  for (std::list<VPlacedVolume *>::const_iterator v = g_volume_list.begin(),
       v_end = g_volume_list.end(); v != v_end; ++v) {
    if ((*v)->id() == id) return *v;
  }
  return NULL;
}

VPlacedVolume* VPlacedVolume::FindVolume(char const *const label) {
  for (std::list<VPlacedVolume *>::const_iterator v = g_volume_list.begin(),
       v_end = g_volume_list.end(); v != v_end; ++v) {
    if ((*v)->label() == label) return *v;
  }
  return NULL;
}

} // End global namespace