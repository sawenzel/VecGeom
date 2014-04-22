/**
 * @file placed_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <stdio.h>

#include "volumes/placed_volume.h"
#include "volumes/placed_box.h"

namespace VECGEOM_NAMESPACE {

int VPlacedVolume::g_id_count = 0;
std::list<VPlacedVolume *> VPlacedVolume::g_volume_list =
    std::list<VPlacedVolume *>();

VPlacedVolume::~VPlacedVolume() {
  delete label_;
  g_volume_list.remove(this);
}

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::set_bounding_box(VPlacedVolume const *const bbox) {
  bounding_box_ = dynamic_cast<PlacedBox const*>(bbox);
}

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::Print(const int indent) const {
  for (int i = 0; i < indent; ++i) printf("  ");
  PrintType();
  printf(" [%i]", id_);
#ifndef VECGEOM_NVCC
  if (label_->size()) {
    printf(" \"%s\"", label_->c_str());
  }
#endif
  printf(": \n");
  for (int i = 0; i <= indent; ++i) printf("  ");
  matrix_->Print();
  printf("\n");
  logical_volume_->Print(indent+1);
}

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::PrintContent(const int indent) const {
  Print(indent);
  if( daughters().size() > 0){
    printf(":");
    for (Iterator<VPlacedVolume const*> vol = daughters().begin();
         vol != daughters().end(); ++vol) {
      printf("\n");
      (*vol)->PrintContent(indent+3);
    }
  }
}

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol) {
  os << "(" << (*vol.unplaced_volume()) << ", " << (*vol.matrix()) << ")";
  return os;
}

} // End global namespace
